using QRMumps, SparseMatricesCOO, LinearOperators, Krylov
export QRMumpsSubsolver, LSMRSubsolver, LSQRSubsolver, CGLSSubsolver
export AbstractR2NLSSubsolver, KrylovR2NLSSubsolver, QRMumpsR2NLSSubsolver

# ==============================================================================
#  QRMumps Subsolver (Aligned with HSLR2NSubsolver style)
# ==============================================================================

"""
    DEFAULT_QRMUMPS_DENSE_MAX_ENTRIES

Default `dense_max_entries` of [`QRMumpsR2NLSSubsolver`](@ref): the largest dense
Jacobian, in number of entries, still handed to qr_mumps. Set to
`DEFAULT_HSL_DENSE_MAX_NVAR^2`, i.e. a dense `12_000 × 12_000` Jacobian, so both
direct subsolvers accept dense problems of comparable size.

Unlike the Hessian, the Jacobian is rectangular, so the threshold is on `m * n`
rather than on `nvar` alone: a dense `m = 1_000_000`, `n = 200` Jacobian is
2 × 10⁸ entries and is caught even though `n` is small.
"""
const DEFAULT_QRMUMPS_DENSE_MAX_ENTRIES = DEFAULT_HSL_DENSE_MAX_NVAR^2

"""
    dense_jacobian_entries(m, n)

Number of entries of a dense `m × n` Jacobian, `m * n`.
"""
dense_jacobian_entries(m::Int, n::Int) = m * n

# Cheap O(1) size and density guard. Below `dense_max_entries` even a fully dense
# Jacobian is cheap enough to factorize, so the density test only arms above it.
function qrmumps_guard_triggered(m, n, nnzj, fill_ratio, dense_max_entries, max_nvar, max_nnzj)
  n > max_nvar && return true
  nnzj > max_nnzj && return true
  dense_nnz = dense_jacobian_entries(m, n)
  return dense_nnz > dense_max_entries && nnzj > fill_ratio * dense_nnz
end

"""
    QRMumpsR2NLSSubsolver(nls; fill_ratio = 0.5,
                          dense_max_entries = 12_000^2,
                          max_nvar = typemax(Int), max_nnzj = typemax(Int))

Direct sparse subsolver for [`R2NLS`](@ref) backed by qr_mumps.

Before touching qr_mumps, a cheap input-size and density guard is applied. If any of

- `nls.meta.nvar > max_nvar`,
- `nls.nls_meta.nnzj > max_nnzj` (`nnzj` counts stored Jacobian nonzeros), or
- `nnzj > fill_ratio * m * n` when `m * n > dense_max_entries`

is true, the constructor returns an "unsupported" placeholder: no qr_mumps object
is allocated, no symbolic analysis is run, and [`is_unsupported`](@ref) returns
`true`. `R2NLS` then short-circuits to status `:exception`. Use this to avoid the
eager symbolic analysis on problems that are too big to factorize.

# Density guard and problem size

A dense Jacobian is cheap enough to factorize at moderate size, so the
`fill_ratio` density test only applies once `m * n > dense_max_entries`: any
problem with `m * n ≤ dense_max_entries` is handed to qr_mumps no matter how
dense it is. The threshold is on entries rather than on `nvar` because the
Jacobian is `m × n`: a dense `m = 1_000_000`, `n = 200` Jacobian is 2 × 10⁸
entries (~1.6 GB in `Float64`) and must be caught even though `n` is small.

With the default `dense_max_entries = 12_000^2`, a dense `m = n = 5_000` problem
still runs through qr_mumps. Set it to the largest dense Jacobian you accept
factorizing; `dense_max_entries = 0` restores the "reject every dense Jacobian"
behaviour, while `max_nvar`/`max_nnzj` stay hard caps that apply regardless of
density.

Negative `max_nvar`, `max_nnzj` or `dense_max_entries` raise `ArgumentError`.
"""
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
  dense_max_entries::Int # largest dense Jacobian, in entries, still accepted
  max_nvar::Int     # hard cap on nvar used by the guard
  max_nnzj::Int     # hard cap on nnzj used by the guard
  unsupported::Bool # true when the Jacobian is too large or dense for the direct solver

  function QRMumpsR2NLSSubsolver(
    nls::AbstractNLSModel{T};
    fill_ratio::Real = 0.5,
    dense_max_entries::Int = DEFAULT_QRMUMPS_DENSE_MAX_ENTRIES,
    max_nvar::Int = typemax(Int),
    max_nnzj::Int = typemax(Int),
  ) where {T}
    max_nvar >= 0 || throw(ArgumentError("max_nvar must be nonnegative"))
    max_nnzj >= 0 || throw(ArgumentError("max_nnzj must be nonnegative"))
    dense_max_entries >= 0 || throw(ArgumentError("dense_max_entries must be nonnegative"))
    meta = nls.meta
    n = meta.nvar
    m = nls.nls_meta.nequ
    nnzj = nls.nls_meta.nnzj
    fr = T(fill_ratio)

    # Skip building the QRMumps object (and its eager symbolic analysis) when the
    # problem is too large, or dense *and* larger than `dense_max_entries`.
    if qrmumps_guard_triggered(m, n, nnzj, fr, dense_max_entries, max_nvar, max_nnzj)
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
        dense_max_entries,
        max_nvar,
        max_nnzj,
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

    new{T}(
      spmat,
      spfct,
      irn,
      jcn,
      val,
      b_aug,
      m,
      n,
      nnzj,
      Jx,
      fr,
      dense_max_entries,
      max_nvar,
      max_nnzj,
      false,
    )
  end
end

"""
    QRMumpsSubsolver(nls; fill_ratio = 0.5, dense_max_entries = 12_000^2,
                     max_nvar = typemax(Int), max_nnzj = typemax(Int))

[`QRMumpsR2NLSSubsolver`](@ref) constructor. Same guards and keyword semantics.
"""
QRMumpsSubsolver(
  nls;
  fill_ratio::Real = 0.5,
  dense_max_entries::Int = DEFAULT_QRMUMPS_DENSE_MAX_ENTRIES,
  max_nvar::Int = typemax(Int),
  max_nnzj::Int = typemax(Int),
) = QRMumpsR2NLSSubsolver(
  nls;
  fill_ratio = fill_ratio,
  dense_max_entries = dense_max_entries,
  max_nvar = max_nvar,
  max_nnzj = max_nnzj,
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