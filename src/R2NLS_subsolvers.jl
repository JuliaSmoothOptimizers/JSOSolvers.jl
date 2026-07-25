using QRMumps, SparseMatricesCOO, LinearOperators, Krylov
export QRMumpsSubsolver, LSMRSubsolver, LSQRSubsolver, CGLSSubsolver
export AbstractR2NLSSubsolver, KrylovR2NLSSubsolver, QRMumpsR2NLSSubsolver

# ==============================================================================
#  QRMumps Subsolver (Aligned with HSLR2NSubsolver style)
# ==============================================================================

mutable struct QRMumpsR2NLSSubsolver{T} <: AbstractR2NLSSubsolver{T}
  spmat::Union{qrm_spmat{T}, Nothing}
  spfct::Union{qrm_spfct{T}, Nothing}
  irn::Vector{Int}
  jcn::Vector{Int}
  val::Vector{T}
  b_aug::Vector{T}
  m::Int
  n::Int
  nnzj::Int
  Jx::SparseMatrixCOO{T, Int}
  fill_ratio::T     # density threshold used by the guard
  min_matrix_size::Int # matrix entry threshold used by the guard
  unsupported::Bool # true when the Jacobian is too dense for the direct solver

  function QRMumpsR2NLSSubsolver(
    nls::AbstractNLSModel{T};
    fill_ratio::Real = 0.5,
    min_matrix_size::Int = 1_000,
  ) where {T}
    meta = nls.meta
    n = meta.nvar
    m = nls.nls_meta.nequ
    nnzj = nls.nls_meta.nnzj
    fr = T(fill_ratio)

    # Cheap O(1) size and density guard. Large, dense Jacobians can make the eager
    # QRMumps symbolic analysis prohibitively expensive, so skip building the
    # QRMumps object in that case.
    dense_nnz = m * n
    if dense_nnz > min_matrix_size && nnzj > fr * dense_nnz
      return new{T}(
        nothing,
        nothing,
        Int[],
        Int[],
        T[],
        T[],
        m,
        n,
        nnzj,
        SparseMatrixCOO(m, n, Int[], Int[], T[]),
        fr,
        min_matrix_size,
        true,
      )
    end

    qrm_init()

    irn = Vector{Int}(undef, nnzj + n)
    jcn = Vector{Int}(undef, nnzj + n)
    val = Vector{T}(undef, nnzj + n)

    jac_structure_residual!(nls, view(irn, 1:nnzj), view(jcn, 1:nnzj))

    @inbounds for i = 1:n
      irn[nnzj + i] = m + i
      jcn[nnzj + i] = i
    end

    Jx = SparseMatrixCOO(m, n, irn[1:nnzj], jcn[1:nnzj], val[1:nnzj])

    spmat = qrm_spmat_init(m + n, n, irn, jcn, val; sym = false)
    spfct = qrm_spfct_init(spmat)
    b_aug = Vector{T}(undef, m + n)

    qrm_analyse!(spmat, spfct; transp = 'n')

    new{T}(spmat, spfct, irn, jcn, val, b_aug, m, n, nnzj, Jx, fr, min_matrix_size, false)
  end
end

QRMumpsSubsolver(nls; fill_ratio::Real = 0.5, min_matrix_size::Int = 1_000) =
  QRMumpsR2NLSSubsolver(
    nls;
    fill_ratio = fill_ratio,
    min_matrix_size = min_matrix_size,
  )

is_unsupported(sub::QRMumpsR2NLSSubsolver) = sub.unsupported

function initialize!(sub::QRMumpsR2NLSSubsolver, nls, x)
  update_subsolver!(sub, nls, x)
  return nothing
end

function update_subsolver!(sub::QRMumpsR2NLSSubsolver, nls, x)
  jac_coord_residual!(nls, x, view(sub.val, 1:sub.nnzj))
  sub.Jx.vals .= view(sub.val, 1:sub.nnzj)
  return nothing
end

function (sub::QRMumpsR2NLSSubsolver{T})(s, rhs, σ, atol, rtol, n; verbose = 0) where {T}
  sqrt_σ = sqrt(σ)

  @inbounds for i = 1:n
    sub.val[sub.nnzj + i] = sqrt_σ
  end

  sub.b_aug[1:sub.m] .= rhs
  sub.b_aug[(sub.m + 1):end] .= zero(T)

  qrm_factorize!(sub.spmat, sub.spfct; transp = 'n')
  qrm_apply!(sub.spfct, sub.b_aug; transp = 't')
  qrm_solve!(sub.spfct, sub.b_aug, s; transp = 'n')

  return true, :solved, 1
end

get_jacobian(sub::QRMumpsR2NLSSubsolver) = sub.Jx
get_operator_norm(sub::QRMumpsR2NLSSubsolver) = norm(sub.Jx.vals)

# ==============================================================================
#  Krylov Subsolvers (Aligned with KrylovR2NSubsolver style)
# ==============================================================================

mutable struct KrylovR2NLSSubsolver{T, V, Op, W} <: AbstractR2NLSSubsolver{T}
  workspace::W
  Jx::Op
  solver_name::Symbol
  Jv::V    
  Jtv::V   

  function KrylovR2NLSSubsolver(nls::AbstractNLSModel{T, V}, solver_name::Symbol) where {T, V}
    m = nls.nls_meta.nequ
    n = nls.meta.nvar

    Jv = V(undef, m)
    Jtv = V(undef, n)
    Jx = jac_op_residual!(nls, nls.meta.x0, Jv, Jtv)

    workspace = krylov_workspace(Val(solver_name), m, n, V)
    
    # THE FIX: All 5 parameters are now correctly passed to new()
    new{T, V, typeof(Jx), typeof(workspace)}(workspace, Jx, solver_name, Jv, Jtv)
  end
end

LSMRSubsolver(nls) = KrylovR2NLSSubsolver(nls, :lsmr)
LSQRSubsolver(nls) = KrylovR2NLSSubsolver(nls, :lsqr)
CGLSSubsolver(nls) = KrylovR2NLSSubsolver(nls, :cgls)

function initialize!(sub::KrylovR2NLSSubsolver, nls, x)
  # Because Jv and Jtv were safely initialized in the constructor, this will no longer crash
  sub.Jx = jac_op_residual!(nls, x, sub.Jv, sub.Jtv)
  return nothing
end

function update_subsolver!(sub::KrylovR2NLSSubsolver, nls, x)
  return nothing
end

function (sub::KrylovR2NLSSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  sub.workspace.stats.niter = 0

  krylov_solve!(
    sub.workspace,
    sub.Jx,
    rhs,
    atol = atol,
    rtol = rtol,
    λ = sqrt(σ),
    # itmax = max(2 * (size(sub.Jx, 1) + size(sub.Jx, 2)), 50),
    itmax = max(2 * n, 50),
    verbose = verbose,
  )
  
  s .= sub.workspace.x
  return Krylov.issolved(sub.workspace), sub.workspace.stats.status, sub.workspace.stats.niter
end

get_jacobian(sub::KrylovR2NLSSubsolver) = sub.Jx

function get_operator_norm(sub::KrylovR2NLSSubsolver)
  λmax, _ = LinearOperators.estimate_opnorm(sub.Jx)
  return λmax
end