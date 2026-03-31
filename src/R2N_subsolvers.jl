using HSL
export ShiftedLBFGSSolver, HSLR2NSubsolver, KrylovR2NSubsolver
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

  function KrylovR2NSubsolver(nlp::AbstractNLPModel{T, V}, solver_name::Symbol = :cg) where {T, V}
    x_init = nlp.meta.x0
    n = nlp.meta.nvar
    H = hess_op(nlp, x_init)

    A = nothing
    if solver_name in (:cg, :cr)
      A = ShiftedOperator(H)
    end

    workspace = krylov_workspace(Val(solver_name), n, n, V)

    new{T, V, typeof(H), typeof(workspace), typeof(A)}(workspace, H, A, solver_name, V(undef, n))
  end
end

CGR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cg)
CRR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cr)
MinresR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres)
MinresQlpR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres_qlp)

function initialize!(sub::KrylovR2NSubsolver, nlp, x)
  return nothing
end

function update_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  # Standard hess_op updates internally if it holds the NLP reference
  return nothing
end

function (sub::KrylovR2NSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  sub.workspace.stats.niter = 0

  if sub.solver_name in (:cg, :cr)
    sub.A.σ = σ
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
  else # minres, minres_qlp
    krylov_solve!(
      sub.workspace,
      sub.H,
      rhs,
      λ = σ,
      itmax = max(2 * n, 50),
      atol = atol,
      rtol = rtol,
      verbose = verbose,
      linesearch = true,
    )
  end

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

get_operator(sub::KrylovR2NSubsolver) = sub.H
has_npc_direction(sub::KrylovR2NSubsolver) =
  isdefined(sub.workspace, :npc_dir) && sub.workspace.stats.npcCount > 0

function get_npc_direction(sub::KrylovR2NSubsolver)
  has_npc_direction(sub) || error("No NPC direction found.")
  return sub.npc_dir
end
function get_operator_norm(sub::KrylovR2NSubsolver)
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

ShiftedLBFGSSolver(nlp) = ShiftedLBFGSSolver(nlp)

initialize!(sub::ShiftedLBFGSSolver, nlp, x) = nothing
update_subsolver!(sub::ShiftedLBFGSSolver, nlp, x) = nothing # LBFGS updates via push! in outer loop

function (sub::ShiftedLBFGSSolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  # rhs is usually -∇f. solve_shifted_system! expects negative gradient
  solve_shifted_system!(s, sub.H, rhs, σ)
  return true, :first_order, 1, 0
end

get_operator(sub::ShiftedLBFGSSolver) = sub.H

function get_operator_norm(sub::ShiftedLBFGSSolver)
  # Estimate norm of H. 
  val, _ = LinearOperators.estimate_opnorm(sub.H)
  return val
end

# ==============================================================================
#   HSL Subsolver (MA97 / MA57)
# ==============================================================================

mutable struct HSLR2NSubsolver{T, S} <: AbstractR2NSubsolver{T}
  hsl_obj::S
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector{T}
  n::Int
  nnzh::Int
  work::Vector{T} # workspace for solves (used for MA57)
end

function HSLR2NSubsolver(nlp::AbstractNLPModel{T, V}; hsl_constructor = ma97_coord) where {T, V}
  LIBHSL_isfunctional() || error("HSL library is not functional")
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  total_nnz = nnzh + n

  rows = Vector{Int}(undef, total_nnz)
  cols = Vector{Int}(undef, total_nnz)
  vals = Vector{T}(undef, total_nnz)

  # Structure analysis must happen in constructor to define the object type S
  hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))

  # Initialize values to zero. Actual computation happens in initialize!
  fill!(vals, zero(T))

  @inbounds for i = 1:n
    rows[nnzh + i] = i
    cols[nnzh + i] = i
    # Diagonal shift will be updated during solve using σ
    vals[nnzh + i] = one(T)
  end

  hsl_obj = hsl_constructor(n, cols, rows, vals)

  if hsl_constructor == ma57_coord
    work = Vector{T}(undef, n * size(nlp.meta.x0, 2))
  else
    work = Vector{T}(undef, 0)
  end

  return HSLR2NSubsolver{T, typeof(hsl_obj)}(hsl_obj, rows, cols, vals, n, nnzh, work)
end

MA97R2NSubsolver(nlp) = HSLR2NSubsolver(nlp; hsl_constructor = ma97_coord)
MA57R2NSubsolver(nlp) = HSLR2NSubsolver(nlp; hsl_constructor = ma57_coord)

function initialize!(sub::HSLR2NSubsolver, nlp, x)
  # Compute the initial Hessian values at x
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return nothing
end

function update_subsolver!(sub::HSLR2NSubsolver, nlp, x)
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
end

function get_inertia(sub::HSLR2NSubsolver{T, S}) where {T, S <: Ma97{T}}
  n = sub.n
  num_neg = sub.hsl_obj.info.num_neg
  num_zero = n - sub.hsl_obj.info.matrix_rank
  return num_neg, num_zero
end

function get_inertia(sub::HSLR2NSubsolver{T, S}) where {T, S <: Ma57{T}}
  n = sub.n
  num_neg = sub.hsl_obj.info.num_negative_eigs
  num_zero = n - sub.hsl_obj.info.rank
  return num_neg, num_zero
end

function _hsl_factor_and_solve!(sub::HSLR2NSubsolver{T, S}, g, s) where {T, S <: Ma97{T}}
  ma97_factorize!(sub.hsl_obj)
  if sub.hsl_obj.info.flag < 0
    return false, :err, 0, 0
  end
  s .= g
  ma97_solve!(sub.hsl_obj, s)
  return true, :first_order, 1, 0
end

function _hsl_factor_and_solve!(sub::HSLR2NSubsolver{T, S}, g, s) where {T, S <: Ma57{T}}
  ma57_factorize!(sub.hsl_obj)
  s .= g
  ma57_solve!(sub.hsl_obj, s, sub.work)
  return true, :first_order, 1, 0
end

function (sub::HSLR2NSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  # Update diagonal shift in the vals array
  @inbounds for i = 1:n
    sub.vals[sub.nnzh + i] = σ
  end
  return _hsl_factor_and_solve!(sub, rhs, s)
end

get_operator(sub::HSLR2NSubsolver) = sub

function get_operator_norm(sub::HSLR2NSubsolver)
  # Cheap estimate of norm using the stored values
  # Exclude the shift values (last n elements) which are at indices nnzh+1:end
  return norm(view(sub.vals, 1:sub.nnzh), Inf)
end

# Helper to support `mul!` for HSL subsolver
function LinearAlgebra.mul!(y::AbstractVector, sub::HSLR2NSubsolver, x::AbstractVector)
  coo_sym_prod!(
    view(sub.rows, 1:sub.nnzh),
    view(sub.cols, 1:sub.nnzh),
    view(sub.vals, 1:sub.nnzh),
    x,
    y,
  )
end
