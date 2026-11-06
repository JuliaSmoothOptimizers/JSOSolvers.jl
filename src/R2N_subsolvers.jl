export ShiftedLBFGSSolver, KrylovR2NSubsolver
export CGR2NSubsolver, CRR2NSubsolver, MinresR2NSubsolver, MinresQlpR2NSubsolver
export AbstractR2NSubsolver
export MA97R2NSubsolver, MA57R2NSubsolver

# ==============================================================================
#   Krylov Subsolver (CG, CR, MINRES)
# ==============================================================================

mutable struct KrylovR2NSubsolver{T, V, Op, W, ShiftOp} <: AbstractR2NSubsolver{T}
  workspace::W
  H::Op           # The Hessian Operator
  A::ShiftOp      # The Shifted Operator (only for CG/CR)
  solver_name::Symbol
  npc_dir::V      # Store NPC direction if needed
  x::V            # Own copy of the iterate; H is evaluated at this vector

  function KrylovR2NSubsolver(nlp::AbstractNLPModel{T, V}, solver_name::Symbol = :cg) where {T, V}
    n = nlp.meta.nvar
    x = copy(nlp.meta.x0)
    H = isa(nlp, LBFGSModel) ? nlp.op : hess_op(nlp, x)
    A = ShiftedOperator(H)

    workspace = krylov_workspace(Val(solver_name), n, n, V)

    new{T, V, typeof(H), typeof(workspace), typeof(A)}(workspace, H, A, solver_name, V(undef, n), x)
  end
end

CGR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cg)
CRR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cr)
MinresR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres)
MinresQlpR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres_qlp)

# The operators are built once on sub.x and follow it in place, so solve! does not
# allocate; they are only rebuilt by reset_subsolver! for a new model.
function initialize!(sub::KrylovR2NSubsolver, nlp, x)
  sub.x .= x
  return sub
end

function update_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  sub.x .= x
  return sub
end

function reset_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  sub.x .= x
  sub.H = isa(nlp, LBFGSModel) ? nlp.op : hess_op(nlp, sub.x)
  sub.A = ShiftedOperator(sub.H)
  return sub
end

function (sub::KrylovR2NSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  sub.workspace.stats.niter = 0

  sub.A.data.σ = σ
  krylov_solve!(
    sub.workspace,
    sub.A,
    rhs,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = rtol,
    verbose = verbose,
    linesearch = true,
  )

  s .= sub.workspace.x
  if isdefined(sub.workspace, :npc_dir)
    sub.npc_dir .= sub.workspace.npc_dir
  end

  # Return the tuple expected by the main loop
  return Krylov.issolved(sub.workspace),
  sub.workspace.stats.status,
  sub.workspace.stats.niter,
  sub.workspace.stats.npcCount
end

get_model_hessian(sub::KrylovR2NSubsolver) = sub.H
has_npc_direction(sub::KrylovR2NSubsolver) =
  isdefined(sub.workspace, :npc_dir) && sub.workspace.stats.npcCount > 0

function get_npc_direction(sub::KrylovR2NSubsolver)
  has_npc_direction(sub) || error("No NPC direction found.")
  return sub.npc_dir
end
function get_model_hessian_norm(sub::KrylovR2NSubsolver)
  # Estimate norm of H.
  val, _ = LinearOperators.estimate_opnorm(sub.H)
  return val
end

# ==============================================================================
#   Shifted LBFGS Subsolver
# ==============================================================================

mutable struct ShiftedLBFGSSolver{T, Op} <: AbstractR2NSubsolver{T}
  H::Op # The LBFGS Operator

  function ShiftedLBFGSSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
    if !(nlp isa LBFGSModel)
      error("ShiftedLBFGSSolver can only be used by LBFGSModel")
    end
    new{T, typeof(nlp.op)}(nlp.op)
  end
end

initialize!(sub::ShiftedLBFGSSolver, nlp, x) = sub
update_subsolver!(sub::ShiftedLBFGSSolver, nlp, x) = sub # LBFGS updates via push! in outer loop

# A new model brings its own LBFGS operator; the generic method would keep the old one.
function reset_subsolver!(sub::ShiftedLBFGSSolver, nlp, x)
  nlp isa LBFGSModel || error("ShiftedLBFGSSolver can only be used by LBFGSModel")
  sub.H = nlp.op
  return sub
end

function (sub::ShiftedLBFGSSolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  # rhs is usually -∇f. solve_shifted_system! expects negative gradient
  solve_shifted_system!(s, sub.H, rhs, σ)
  return true, :first_order, 1, 0
end

get_model_hessian(sub::ShiftedLBFGSSolver) = sub.H

function get_model_hessian_norm(sub::ShiftedLBFGSSolver)
  # Estimate norm of H.
  val, _ = LinearOperators.estimate_opnorm(sub.H)
  return val
end

# ==============================================================================
#   HSL Subsolver (MA97 / MA57)
# ==============================================================================

"""
    DEFAULT_HSL_DENSE_MAX_NVAR

Default `dense_max_nvar` of [`MA57R2NSubsolver`](@ref) and [`MA97R2NSubsolver`](@ref): the largest `nvar` for
which a fully dense Hessian is still handed to HSL.
"""
const DEFAULT_HSL_DENSE_MAX_NVAR = 12_000

"""
    dense_hessian_entries(n)

Number of stored entries of a dense symmetric `n × n` Hessian, `n(n+1)/2`.
"""
dense_hessian_entries(n::Int) = (n * (n + 1)) ÷ 2

# Cheap O(1) size and density guard shared by the constructor and
# `reset_subsolver!`. Below `dense_max_nvar` even a fully dense Hessian is cheap
# enough to factorize, so the density test only arms above that size.
function hsl_guard_triggered(n, nnzh, fill_ratio, dense_max_nvar, max_nvar, max_nnzh)
  n > max_nvar && return true
  nnzh > max_nnzh && return true
  return n > dense_max_nvar && nnzh > fill_ratio * dense_hessian_entries(n)
end

"""
    MA97R2NSubsolver(nlp; fill_ratio = 0.5, dense_max_nvar = 12_000,
                     max_nvar = typemax(Int), max_nnzh = typemax(Int))
    MA57R2NSubsolver(nlp; fill_ratio = 0.5, dense_max_nvar = 12_000,
                     max_nvar = typemax(Int), max_nnzh = typemax(Int))

Direct sparse subsolvers for [`R2N`](@ref) backed by HSL's MA97 and MA57. They are only
available after `using HSL`, and running them requires a licensed HSL_jll.

Before touching HSL, a cheap input-size and density guard is applied. If any of

- `nlp.meta.nvar > max_nvar`,
- `nlp.meta.nnzh > max_nnzh` (`nnzh` counts stored lower-triangular Hessian nonzeros), or
- `nnzh > fill_ratio * n(n+1)/2` when `n > dense_max_nvar`

is true, the constructor returns an "unsupported" placeholder: no HSL object is
allocated, no symbolic analysis is run, and [`is_unsupported`](@ref) returns `true`.
`R2N` then short-circuits to status `:exception`. Use this to avoid HSL's eager
symbolic analysis on problems that are too big to factorize (e.g. `n ≈ 120_000`).

# Density guard and problem size

A dense Hessian is cheap enough to factorize at moderate `n`, so the `fill_ratio`
density test only applies once `n > dense_max_nvar`: any problem with
`n ≤ dense_max_nvar` is handed to HSL no matter how dense it is. With the default
`dense_max_nvar = 12_000`, a dense `n = 5_000` problem still runs through HSL.

Set `dense_max_nvar` to the largest `n` for which you accept factorizing a dense
Hessian (work grows like `n³`, storage like `n²`). `dense_max_nvar = 0` restores
the "reject every dense Hessian" behaviour, while `max_nvar`/`max_nnzh` stay hard
caps that apply regardless of density.
"""
function MA97R2NSubsolver end
@doc (@doc MA97R2NSubsolver) function MA57R2NSubsolver end
