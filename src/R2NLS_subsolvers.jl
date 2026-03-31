using QRMumps, SparseMatricesCOO, LinearOperators
export QRMumpsSubsolver, LSMRSubsolver, LSQRSubsolver, CGLSSubsolver

# ==============================================================================
#  QRMumps Subsolver
# ==============================================================================

mutable struct QRMumpsSubsolver{T} <: AbstractR2NLSSubsolver{T}
  spmat::qrm_spmat{T}
  spfct::qrm_spfct{T}
  irn::Vector{Int}
  jcn::Vector{Int}
  val::Vector{T}
  b_aug::Vector{T}
  m::Int
  n::Int
  nnzj::Int

  # Stored internally, initialized in constructor
  Jx::SparseMatrixCOO{T, Int}

  function QRMumpsSubsolver(nls::AbstractNLSModel{T}) where {T}
    qrm_init()
    meta = nls.meta
    n = meta.nvar
    m = nls.nls_meta.nequ
    nnzj = nls.nls_meta.nnzj

    # 1. Allocate Arrays
    irn = Vector{Int}(undef, nnzj + n)
    jcn = Vector{Int}(undef, nnzj + n)
    val = Vector{T}(undef, nnzj + n)

    # 2. FILL STRUCTURE IMMEDIATELY 
    # This populates the first nnzj elements with the Jacobian pattern
    jac_structure_residual!(nls, view(irn, 1:nnzj), view(jcn, 1:nnzj))

    # 3. Fill Regularization Structure
    # Rows m+1 to m+n, Columns 1 to n
    @inbounds for i = 1:n
      irn[nnzj + i] = m + i
      jcn[nnzj + i] = i
    end

    # 4. Create Jx
    # Since irn/jcn are already populated, Jx is valid immediately.
    # It copies the structure from irn/jcn.
    Jx = SparseMatrixCOO(m, n, irn[1:nnzj], jcn[1:nnzj], val[1:nnzj])

    # 5. Initialize QRMumps
    spmat = qrm_spmat_init(m + n, n, irn, jcn, val; sym = false)
    spfct = qrm_spfct_init(spmat)
    b_aug = Vector{T}(undef, m + n)

    # 6. Analyze Sparsity
    qrm_analyse!(spmat, spfct; transp = 'n')

    sub = new{T}(spmat, spfct, irn, jcn, val, b_aug, m, n, nnzj, Jx)
    return sub
  end
end

function initialize!(sub::QRMumpsSubsolver, nls, x)
  # Just update the values for the new x.
  update_subsolver!(sub, nls, x)
end

function update_subsolver!(sub::QRMumpsSubsolver, nls, x)
  # 1. Compute Jacobian values into QRMumps 'val' array
  jac_coord_residual!(nls, x, view(sub.val, 1:sub.nnzj))

  # 2. Explicitly sync to Jx (Copy values)
  # This ensures Jx.vals has the fresh Jacobian for the gradient calculation
  sub.Jx.vals .= view(sub.val, 1:sub.nnzj)
end

function (sub::QRMumpsSubsolver{T})(s, rhs, σ, atol, rtol; verbose = 0) where {T}
  sqrt_σ = sqrt(σ)

  # 1. Update ONLY the regularization values 
  @inbounds for i = 1:sub.n
    sub.val[sub.nnzj + i] = sqrt_σ
  end

  # 2. Tell QRMumps values changed
  qrm_update_subsolver!(sub.spmat, sub.val)

  # 3. Prepare RHS [-F(x); 0]
  sub.b_aug[1:sub.m] .= rhs
  sub.b_aug[(sub.m + 1):end] .= zero(T)

  # 4. Factorize and Solve
  qrm_factorize!(sub.spmat, sub.spfct; transp = 'n')
  qrm_apply!(sub.spfct, sub.b_aug; transp = 't')
  qrm_solve!(sub.spfct, sub.b_aug, s; transp = 'n')

  return true, :solved, 1
end

get_jacobian(sub::QRMumpsSubsolver) = sub.Jx

function get_operator_norm(sub::QRMumpsSubsolver)
  # The Frobenius norm is extremely cheap to compute from the COO values 
  # and serves as a mathematically valid upper bound for the operator 2-norm.
  return norm(sub.Jx.vals)
end

# ==============================================================================
#   Krylov Subsolvers (LSMR, LSQR, CGLS)
# ==============================================================================

mutable struct GenericKrylovSubsolver{T, V, Op, W} <: AbstractR2NLSSubsolver{T}
  workspace::W
  Jx::Op
  solver_name::Symbol

  function GenericKrylovSubsolver(nls::AbstractNLSModel{T, V}, solver_name::Symbol) where {T, V}
    x_init = nls.meta.x0
    m = nls.nls_meta.nequ
    n = nls.meta.nvar

    # Jx and its buffers allocated here inside the subsolver
    Jv = V(undef, m)
    Jtv = V(undef, n)
    Jx = jac_op_residual!(nls, x_init, Jv, Jtv)

    workspace = krylov_workspace(Val(solver_name), m, n, V)
    new{T, V, typeof(Jx), typeof(workspace)}(workspace, Jx, solver_name)
  end
end

# Specific Constructors for Uniform Signature (nls, x)
LSMRSubsolver(nls, x) = GenericKrylovSubsolver(nls, :lsmr)
LSQRSubsolver(nls, x) = GenericKrylovSubsolver(nls, :lsqr)
CGLSSubsolver(nls, x) = GenericKrylovSubsolver(nls, :cgls)

function update_subsolver!(sub::GenericKrylovSubsolver, nls, x)
  # Implicitly updated because Jx holds reference to x.
  # We just ensure x is valid.
  nothing
end

function (sub::GenericKrylovSubsolver)(s, rhs, σ, atol, rtol; verbose = 0)
  # λ allocation/calculation happens here in the solve
  krylov_solve!(
    sub.workspace,
    sub.Jx,
    rhs,
    atol = atol,
    rtol = rtol,
    λ = sqrt(σ), # λ allocated here
    itmax = max(2 * (size(sub.Jx, 1) + size(sub.Jx, 2)), 50),
    verbose = verbose,
  )
  s .= sub.workspace.x
  return Krylov.issolved(sub.workspace), sub.workspace.stats.status, sub.workspace.stats.niter
end

get_jacobian(sub::GenericKrylovSubsolver) = sub.Jx
initialize!(sub::GenericKrylovSubsolver, nls, x) = nothing
function get_operator_norm(sub::GenericKrylovSubsolver)
  # Jx is a LinearOperator, so we can use the specialized estimator
  λmax, _ = LinearOperators.estimate_opnorm(sub.Jx)
  return λmax
end
