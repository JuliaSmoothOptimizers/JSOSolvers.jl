export R2N, R2NSolver, R2NParameterSet
export ShiftedLBFGSSolver, HSLR2NSubsolver, KrylovR2NSubsolver
export CGR2NSubsolver, CRR2NSubsolver, MinresR2NSubsolver, MinresQlpR2NSubsolver
export AbstractR2NSubsolver
export MA97R2NSubsolver, MA57R2NSubsolver

using LinearOperators, LinearAlgebra
using SparseArrays
using HSL

"""
    R2NParameterSet([T=Float64]; θ1, θ2, η1, η2, γ1, γ2, γ3, σmin, non_mono_size)

Parameter set for the R2N solver. Controls algorithmic tolerances and step acceptance.

# Keyword Arguments
- `θ1 = T(0.5)`: Cauchy step parameter (0 < θ1 < 1).
- `θ2 = eps(T)^(-1)`: Maximum allowed ratio between the step and the Cauchy step (θ2 > 1).
- `η1 = eps(T)^(1/4)`: Accept step if actual/predicted reduction ≥ η1 (0 < η1 ≤ η2 < 1).
- `η2 = T(0.95)`: Step is very successful if reduction ≥ η2 (0 < η1 ≤ η2 < 1).
- `γ1 = T(1.5)`: Regularization increase factor on successful (but not very successful) step (1 < γ1 ≤ γ2).
- `γ2 = T(2.5)`: Regularization increase factor on rejected step (γ1 ≤ γ2).
- `γ3 = T(0.5)`: Regularization decrease factor on very successful step (0 < γ3 ≤ 1).
- `δ1 = T(0.5)`: Cauchy point calculation parameter.
- `σmin = eps(T)`: Minimum regularization parameter.
- `non_mono_size = 1`: Window size for non-monotone acceptance.
"""
struct R2NParameterSet{T} <: AbstractParameterSet
  θ1::Parameter{T, RealInterval{T}}
  θ2::Parameter{T, RealInterval{T}}
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  γ3::Parameter{T, RealInterval{T}}
  δ1::Parameter{T, RealInterval{T}}
  σmin::Parameter{T, RealInterval{T}}
  non_mono_size::Parameter{Int, IntegerRange{Int}}
  ls_c::Parameter{T, RealInterval{T}}
  ls_increase::Parameter{T, RealInterval{T}}
  ls_decrease::Parameter{T, RealInterval{T}}
  ls_min_alpha::Parameter{T, RealInterval{T}}
  ls_max_alpha::Parameter{T, RealInterval{T}}
end

# Default parameter values
const R2N_θ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_θ2 = DefaultParameter(nlp -> inv(eps(eltype(nlp.meta.x0))), "eps(T)^(-1)")
const R2N_η1 = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))^(T(1) / T(4))
end, "eps(T)^(1/4)")
const R2N_η2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.95), "T(0.95)")
const R2N_γ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1.5), "T(1.5)")
const R2N_γ2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(2.5), "T(2.5)")
const R2N_γ3 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_δ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_σmin = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))
end, "eps(T)")
const R2N_non_mono_size = DefaultParameter(1)
# Line search parameters
const R2N_ls_c = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1e-4), "T(1e-4)")
const R2N_ls_increase = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1.5), "T(1.5)")
const R2N_ls_decrease = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_ls_min_alpha = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1e-8), "T(1e-8)")
const R2N_ls_max_alpha = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1e2), "T(1e2)")

function R2NParameterSet(
  nlp::AbstractNLPModel;
  θ1::T = get(R2N_θ1, nlp),
  θ2::T = get(R2N_θ2, nlp),
  η1::T = get(R2N_η1, nlp),
  η2::T = get(R2N_η2, nlp),
  γ1::T = get(R2N_γ1, nlp),
  γ2::T = get(R2N_γ2, nlp),
  γ3::T = get(R2N_γ3, nlp),
  δ1::T = get(R2N_δ1, nlp),
  σmin::T = get(R2N_σmin, nlp),
  non_mono_size::Int = get(R2N_non_mono_size, nlp),
  ls_c::T = get(R2N_ls_c, nlp),
  ls_increase::T = get(R2N_ls_increase, nlp),
  ls_decrease::T = get(R2N_ls_decrease, nlp),
  ls_min_alpha::T = get(R2N_ls_min_alpha, nlp),
  ls_max_alpha::T = get(R2N_ls_max_alpha, nlp),
) where {T}
  @assert zero(T) < θ1 < one(T) "θ1 must satisfy 0 < θ1 < 1"
  @assert θ2 > one(T) "θ2 must satisfy θ2 > 1"
  @assert zero(T) < η1 <= η2 < one(T) "η1, η2 must satisfy 0 < η1 ≤ η2 < 1"
  @assert one(T) < γ1 <= γ2 "γ1, γ2 must satisfy 1 < γ1 ≤ γ2"
  @assert γ3 > zero(T) && γ3 <= one(T) "γ3 must satisfy 0 < γ3 ≤ 1"
  @assert zero(T) < δ1 < one(T) "δ1 must satisfy 0 < δ1 < 1"

  R2NParameterSet{T}(
    Parameter(θ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(θ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(η1, RealInterval(zero(T), one(T), lower_open = false, upper_open = true)),
    Parameter(η2, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(γ1, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ3, RealInterval(zero(T), one(T), lower_open = false, upper_open = true)),
    Parameter(δ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(σmin, RealInterval(zero(T), T(Inf), lower_open = false, upper_open = true)),
    Parameter(non_mono_size, IntegerRange(1, typemax(Int))),
    Parameter(ls_c, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(ls_increase, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(ls_decrease, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(ls_min_alpha, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(ls_max_alpha, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
  )
end

const npc_handler_allowed = [:gs, :sigma, :prev, :cp]

# ==============================================================================
#   Krylov Subsolver (CG, CR, MINRES)
# ==============================================================================

mutable struct KrylovR2NSubsolver{T, V, Op, W, ShiftOp} <: AbstractR2NSubsolver{T}
  workspace::W
  H::Op           # The Hessian Operator
  A::ShiftOp      # The Shifted Operator (only for CG/CR)
  solver_name::Symbol
  npc_dir::V      # Store NPC direction if needed

  function KrylovR2NSubsolver(
    nlp::AbstractNLPModel{T, V},
    x_init::V,
    solver_name::Symbol = :cg,
  ) where {T, V}
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

CGR2NSubsolver(nlp, x) = KrylovR2NSubsolver(nlp, x, :cg)
CRR2NSubsolver(nlp, x) = KrylovR2NSubsolver(nlp, x, :cr)
MinresR2NSubsolver(nlp, x) = KrylovR2NSubsolver(nlp, x, :minres)
MinresQlpR2NSubsolver(nlp, x) = KrylovR2NSubsolver(nlp, x, :minres_qlp)

function initialize_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  
  return nothing
end

function update_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  # Standard hess_op updates internally if it holds the NLP reference
  return nothing
end

function solve_subproblem!(sub::KrylovR2NSubsolver, s, rhs, σ, atol, rtol, n; verbose = 0)
  sub.workspace.stats.niter = 0

  if sub.solver_name in (:cg, :cr)
    sub.A.σ = σ
    krylov_solve!(
      sub.workspace, sub.A, rhs,
      itmax = max(2 * n, 50), atol = atol, rtol = rtol,
      verbose = verbose, linesearch = true
    )
  else # minres, minres_qlp
    krylov_solve!(
      sub.workspace, sub.H, rhs, λ = σ,
      itmax = max(2 * n, 50), atol = atol, rtol = rtol,
      verbose = verbose, linesearch = true
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
         0 # npcCount placeholder if not available in stats
end

get_operator(sub::KrylovR2NSubsolver) = sub.H
get_npc_direction(sub::KrylovR2NSubsolver) = sub.npc_dir

function get_operator_norm(sub::KrylovR2NSubsolver)
    # Estimate norm of H. 
    val, _ =  LinearOperators.estimate_opnorm(sub.H)
    return val
end

# ==============================================================================
#   Shifted LBFGS Subsolver
# ==============================================================================

mutable struct ShiftedLBFGSSolver{T, Op} <: AbstractR2NSubsolver{T}
  H::Op # The LBFGS Operator

  function ShiftedLBFGSSolver(nlp::AbstractNLPModel{T, V}, x::V) where {T, V}
    if !(nlp isa LBFGSModel)
      error("ShiftedLBFGSSolver can only be used by LBFGSModel")
    end
    new{T, typeof(nlp.op)}(nlp.op)
  end
end

ShiftedLBFGSSolver(nlp, x) = ShiftedLBFGSSolver(nlp, x)

initialize_subsolver!(sub::ShiftedLBFGSSolver, nlp, x) = nothing
update_subsolver!(sub::ShiftedLBFGSSolver, nlp, x) = nothing # LBFGS updates via push! in outer loop

function solve_subproblem!(sub::ShiftedLBFGSSolver, s, rhs, σ, atol, rtol, n; verbose = 0)
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

function HSLR2NSubsolver(nlp::AbstractNLPModel{T, V}, x::V; hsl_constructor=ma97_coord) where {T, V}
  LIBHSL_isfunctional() || error("HSL library is not functional")
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  total_nnz = nnzh + n

  rows = Vector{Int}(undef, total_nnz)
  cols = Vector{Int}(undef, total_nnz)
  vals = Vector{T}(undef, total_nnz)

  # Structure analysis must happen in constructor to define the object type S
  hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))
  
  # Initialize values to zero. Actual computation happens in initialize_subsolver!
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

MA97R2NSubsolver(nlp, x) = HSLR2NSubsolver(nlp, x; hsl_constructor=ma97_coord)
MA57R2NSubsolver(nlp, x) = HSLR2NSubsolver(nlp, x; hsl_constructor=ma57_coord)

function initialize_subsolver!(sub::HSLR2NSubsolver, nlp, x)
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

function solve_subproblem!(sub::HSLR2NSubsolver, s, rhs, σ, atol, rtol, n; verbose=0)
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
    coo_sym_prod!(view(sub.rows, 1:sub.nnzh), 
        view(sub.cols, 1:sub.nnzh), 
        view(sub.vals, 1:sub.nnzh), x, y)
end



# ==============================================================================
#   R2N Solver
# ==============================================================================

"""
    R2N(nlp; kwargs...)

A second-order quadratic regularization method for unconstrained optimization (with shifted L-BFGS or shifted Hessian operator).

    min f(x)

For advanced usage, first define a `R2NSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2NSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments
- `params::R2NParameterSet = R2NParameterSet(nlp)`: algorithm parameters, see [`R2NParameterSet`](@ref).
- `η1::T = $(R2N_η1)`: step acceptance parameter, see [`R2NParameterSet`](@ref).
- `η2::T = $(R2N_η2)`: step acceptance parameter, see [`R2NParameterSet`](@ref).
- `θ1::T = $(R2N_θ1)`: Cauchy step parameter, see [`R2NParameterSet`](@ref).
- `θ2::T = $(R2N_θ2)`: Cauchy step parameter, see [`R2NParameterSet`](@ref).
- `γ1::T = $(R2N_γ1)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `γ2::T = $(R2N_γ2)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `γ3::T = $(R2N_γ3)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `δ1::T = $(R2N_δ1)`: Cauchy point calculation parameter, see [`R2NParameterSet`](@ref).
- `σmin::T = $(R2N_σmin)`: minimum step parameter, see [`R2NParameterSet`](@ref).
- `non_mono_size::Int = $(R2N_non_mono_size)`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of evaluations of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver = CGR2NSubsolver`: the subproblem solver type or instance.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver if KrylovWorkspace type is selected.
- `scp_flag::Bool = true`: if true, we compare the norm of the calculate step with `θ2 * norm(scp)`, each iteration, selecting the smaller step.
- `npc_handler::Symbol = :gs`: the non_positive_curve handling strategy.
  - `:gs`: run line-search along NPC with Goldstein conditions.
  - `:sigma`: increase the regularization parameter σ.
  - `:prev`: if subsolver return after first iteration, increase the sigma, but if subsolver return after second iteration, set s_k = s_k^(t-1).
  - `:cp`: set s_k to Cauchy point.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.
- `callback`: function called at each iteration, see [`Callback`](https://jso.dev/JSOSolvers.jl/stable/#Callback) section.

# Examples
```jldoctest; output = false
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2N(nlp)
# output
"Execution stats: first-order stationary"

```

"""
mutable struct R2NSolver{
  T,
  V,
  Sub <: AbstractR2NSubsolver{T},
  M <: AbstractNLPModel{T, V},
} <: AbstractOptimizationSolver
  x::V              # Current iterate x_k
  xt::V             # Trial iterate x_{k+1}
  gx::V             # Gradient ∇f(x)
  rhs::V            # RHS for subsolver (store -∇f)
  y::V              # Difference of gradients y = ∇f_new - ∇f_old
  Hs::V             # Storage for H*s products
  s::V              # Step direction
  scp::V            # Cauchy point step
  obj_vec::V        # History of objective values for non-monotone strategy
  subsolver::Sub    # The subproblem solver
  h::LineModel{T, V, M} # Line search model
  subtol::T         # Current tolerance for the subproblem
  σ::T              # Regularization parameter
  params::R2NParameterSet{T} # Algorithmic parameters
end

function R2NSolver(
  nlp::AbstractNLPModel{T, V};
  η1 = get(R2N_η1, nlp),
  η2 = get(R2N_η2, nlp),
  θ1 = get(R2N_θ1, nlp),
  θ2 = get(R2N_θ2, nlp),
  γ1 = get(R2N_γ1, nlp),
  γ2 = get(R2N_γ2, nlp),
  γ3 = get(R2N_γ3, nlp),
  δ1 = get(R2N_δ1, nlp),
  σmin = get(R2N_σmin, nlp),
  non_mono_size = get(R2N_non_mono_size, nlp),
  subsolver = CGR2NSubsolver, # Default Type
  ls_c = get(R2N_ls_c, nlp),
  ls_increase = get(R2N_ls_increase, nlp),
  ls_decrease = get(R2N_ls_decrease, nlp),
  ls_min_alpha = get(R2N_ls_min_alpha, nlp),
  ls_max_alpha = get(R2N_ls_max_alpha, nlp),
) where {T, V}
  params = R2NParameterSet(
    nlp;
    η1 = η1,
    η2 = η2,
    θ1 = θ1,
    θ2 = θ2,
    γ1 = γ1,
    γ2 = γ2,
    γ3 = γ3,
    δ1 = δ1,
    σmin = σmin,
    non_mono_size = non_mono_size,
    ls_c = ls_c,
    ls_increase = ls_increase,
    ls_decrease = ls_decrease,
    ls_min_alpha = ls_min_alpha,
    ls_max_alpha = ls_max_alpha,
  )
  
  value(params.non_mono_size) >= 1 || error("non_mono_size must be greater than or equal to 1")

  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  rhs = V(undef, nvar)
  y = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0) # y storage
  Hs = V(undef, nvar)
  
  x .= nlp.meta.x0

  if subsolver isa Union{Type, Function}
     sub_inst = subsolver(nlp, x)
  elseif subsolver isa AbstractR2NSubsolver
     sub_inst = subsolver
  else
     error("subsolver must be a Type or an AbstractR2NSubsolver instance")
  end

  if sub_inst isa ShiftedLBFGSSolver && !(nlp isa LBFGSModel)
     error("ShiftedLBFGSSolver can only be used by LBFGSModel")
  end

  σ = zero(T)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  subtol = one(T)
  obj_vec = fill(typemin(T), non_mono_size)
  
  h = LineModel(nlp, x, s)

  return R2NSolver{T, V, typeof(sub_inst), typeof(nlp)}(
    x,
    xt,
    gx,
    rhs,
    y,
    Hs,
    s,
    scp,
    obj_vec,
    sub_inst,
    h,
    subtol,
    σ,
    params,
  )
end

function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  if solver.subsolver isa KrylovR2NSubsolver
      LinearOperators.reset!(solver.subsolver.H)
  end
  return solver
end

function SolverCore.reset!(solver::R2NSolver{T}, nlp::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  if solver.subsolver isa KrylovR2NSubsolver
      LinearOperators.reset!(solver.subsolver.H)
  end
  solver.h = LineModel(nlp, solver.x, solver.s)
  return solver
end

@doc (@doc R2NSolver) function R2N(
  nlp::AbstractNLPModel{T, V};
  η1::Real = get(R2N_η1, nlp),
  η2::Real = get(R2N_η2, nlp),
  θ1::Real = get(R2N_θ1, nlp),
  θ2::Real = get(R2N_θ2, nlp),
  γ1::Real = get(R2N_γ1, nlp),
  γ2::Real = get(R2N_γ2, nlp),
  γ3::Real = get(R2N_γ3, nlp),
  δ1::Real = get(R2N_δ1, nlp),
  σmin::Real = get(R2N_σmin, nlp),
  non_mono_size::Int = get(R2N_non_mono_size, nlp),
  subsolver = CGR2NSubsolver,
  ls_c::Real = get(R2N_ls_c, nlp),
  ls_increase::Real = get(R2N_ls_increase, nlp),
  ls_decrease::Real = get(R2N_ls_decrease, nlp),
  ls_min_alpha::Real = get(R2N_ls_min_alpha, nlp),
  ls_max_alpha::Real = get(R2N_ls_max_alpha, nlp),
  kwargs...,
) where {T, V}
  solver = R2NSolver(
    nlp;
    η1 = convert(T, η1),
    η2 = convert(T, η2),
    θ1 = convert(T, θ1),
    θ2 = convert(T, θ2),
    γ1 = convert(T, γ1),
    γ2 = convert(T, γ2),
    γ3 = convert(T, γ3),
    δ1 = convert(T, δ1),
    σmin = convert(T, σmin),
    non_mono_size = non_mono_size,
    subsolver = subsolver,
    ls_c = convert(T, ls_c),
    ls_increase = convert(T, ls_increase),
    ls_decrease = convert(T, ls_decrease),
    ls_min_alpha = convert(T, ls_min_alpha),
    ls_max_alpha = convert(T, ls_max_alpha),
  )
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.solve!(
  solver::R2NSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  npc_handler::Symbol = :gs,
  scp_flag::Bool = true,
) where {T, V}
  unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")
  npc_handler in npc_handler_allowed || error("npc_handler must be one of $(npc_handler_allowed)")

  SolverCore.reset!(stats)
  params = solver.params
  η1 = value(params.η1)
  η2 = value(params.η2)
  θ1 = value(params.θ1)
  θ2 = value(params.θ2)
  γ1 = value(params.γ1)
  γ2 = value(params.γ2)
  γ3 = value(params.γ3)
  δ1 = value(params.δ1)
  σmin = value(params.σmin)
  non_mono_size = value(params.non_mono_size)

  ls_c = value(params.ls_c)
  ls_increase = value(params.ls_increase)
  ls_decrease = value(params.ls_decrease)
  
  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.meta.nvar
  x = solver.x .= x
  xt = solver.xt
  ∇fk = solver.gx # current gradient
  rhs = solver.rhs # -∇f for subsolver
  y = solver.y    # gradient difference
  s = solver.s
  scp = solver.scp
  Hs = solver.Hs
  σk = solver.σ

  subtol = solver.subtol
  
  initialize_subsolver!(solver.subsolver, nlp, x)
  H = get_operator(solver.subsolver)

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  ρk = zero(T)

  fmin = min(-one(T), f0) / eps(T)
  unbounded = f0 < fmin

  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  
  if optimal
    @info "Optimal point found at initial point"
    @info log_header(
      [:iter, :f, :dual, :σ, :ρ],
      [Int, Float64, Float64, Float64, Float64],
      hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖"),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk])
  end

  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :f, :dual, :norm_s, :σ, :ρ, :sub_iter, :dir, :sub_status],
      [Int, Float64, Float64, Float64, Float64, Int, String, String],
      hdr_override = Dict(
        :f => "f(x)",
        :dual => "‖∇f‖",
        :norm_s => "‖s‖",
        :sub_iter => "subiter",
        :dir => "dir",
        :sub_status => "status",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, 0.0, σk, ρk, 0, " ", " "])
  end

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      unbounded = unbounded,
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
  solver.σ = σk
  solver.subtol = subtol
  
  if solver.subsolver isa ShiftedLBFGSSolver
    scp_flag = false 
  end

  callback(nlp, solver, stats)
  subtol = solver.subtol
  σk = solver.σ

  done = stats.status != :unknown
  ν_k = one(T)
  γ_k = zero(T)
  ft = f0

  # Initialize scope variables
  step_accepted = false
  sub_stats = :unknown
  subiter = 0
  dir_stat = ""

  while !done
    npcCount = 0
    fck_computed = false 

    # Prepare RHS for subsolver (rhs = -∇f)
    @. rhs = -∇fk 
    
    subsolver_solved, sub_stats, subiter, npcCount = solve_subproblem!(
       solver.subsolver, s, rhs, σk, atol, subtol, n; verbose=subsolver_verbose
    )

    if !subsolver_solved && npcCount == 0
      @warn "Subsolver failed to solve the system. Terminating."
      set_status!(stats, :stalled)
      done = true
      break
    end
    
    calc_scp_needed = false
    force_sigma_increase = false
    if solver.subsolver isa HSLR2NSubsolver
      num_neg, num_zero = get_inertia(solver.subsolver)
    
      if num_zero > 0
          force_sigma_increase = true
      end

      if !force_sigma_increase && num_neg > 0
          mul!(Hs, H, s)
          curv_s = dot(s, Hs)
          
          if curv_s < 0
              npcCount = 1
              if npc_handler == :prev
                  npc_handler = :gs  #Force the npc_handler to be gs and not :prev since we can not have that behavior with HSL subsolver
              end
          else
              calc_scp_needed = true
          end
      end
    end

    if !(solver.subsolver isa ShiftedLBFGSSolver) && npcCount >= 1
      if npc_handler == :gs
        npcCount = 0
        dir = get_npc_direction(solver.subsolver)
        
        # Ensure line search model points to current x and dir
        SolverTools.redirect!(solver.h, x, dir)
        f0_val = stats.objective
        dot_gs = dot(∇fk, dir) # dot_gs = ∇f^T * d

        α, ft, nbk, nbG = armijo_goldstein(
          solver.h, f0_val, dot_gs;
          t = one(T), τ₀ = ls_c, τ₁ = 1 - ls_c,
          γ₀ = ls_decrease, γ₁ = ls_increase,
          bk_max = 100, bG_max = 100, verbose = (verbose > 0),
        )
        @. s = α * dir
        fck_computed = true
      elseif npc_handler == :prev
        npcCount = 0
        # s is already populated by solve_subproblem!
      end
    end

    # Compute Cauchy step
    if scp_flag == true || npc_handler == :cp || calc_scp_needed
      mul!(Hs, H, ∇fk)
      dot_gHg = dot(∇fk, Hs)
      σ_norm_g2 = σk * norm_∇fk^2
      γ_k = (dot_gHg + σ_norm_g2) / norm_∇fk^2

      if γ_k > 0
        ν_k = 2*(1-δ1) / (γ_k)
      else
        λmax = get_operator_norm(solver.subsolver)
        ν_k = θ1 / (λmax + σk)
      end

      # scp = - ν_k * ∇f
      @. scp = -ν_k * ∇fk
      
      if npc_handler == :cp && npcCount >= 1
        npcCount = 0
        s .= scp
      elseif norm(s) > θ2 * norm(scp)
        s .= scp
      end
    end

    if force_sigma_increase || (npc_handler == :sigma && npcCount >= 1)
      step_accepted = false
      σk = max(σmin, γ2 * σk)
      solver.σ = σk
      npcCount = 0
    else
      # Compute Model Predicted Reduction
      mul!(Hs, H, s)
      dot_sHs = dot(s, Hs)    # s' (H + σI) s
      dot_gs = dot(s, ∇fk)  # ∇f' s

      # Predicted Reduction: m(0) - m(s) = -g's - 0.5 s'Bs
      ΔTk = -dot_gs - dot_sHs / 2
      
      # Verify that the predicted reduction is positive and numerically significant.
      # This check handles cases where the subsolver returns a poor step (e.g., if
      # npc_handler=:prev reuses a bad step) or if the reduction is dominated by 
      # machine noise relative to the objective value.
      if ΔTk <= eps(T) * max(one(T), abs(stats.objective))
        step_accepted = false
        σk = max(σmin, γ2 * σk)
        solver.σ = σk
      else
        @. xt = x + s

        if !fck_computed
          ft = obj(nlp, xt)
        end

        if non_mono_size > 1
          k = mod(stats.iter, non_mono_size) + 1
          solver.obj_vec[k] = stats.objective
          ft_max = maximum(solver.obj_vec)
          ρk = (ft_max - ft) / (ft_max - stats.objective + ΔTk)
        else
          ρk = (stats.objective - ft) / ΔTk
        end

        step_accepted = ρk >= η1
        if step_accepted
          # Quasi-Newton Update: Needs ∇f_new - ∇f_old.
          # We have ∇f_old in ∇fk. We need to save it or use a temp.
          # We use `rhs` as temp storage for ∇f_old since we are done with it for this iter.
          if isa(nlp, QuasiNewtonModel)
             rhs .= ∇fk # Save old gradient
          end
          
          x .= xt
          grad!(nlp, x, ∇fk) # ∇fk is now NEW gradient
          
          if isa(nlp, QuasiNewtonModel)
             @. y = ∇fk - rhs # y = new - old
             push!(nlp, s, y)
          end
          
          if !(solver.subsolver isa ShiftedLBFGSSolver)
               update_subsolver!(solver.subsolver, nlp, x) 
               H = get_operator(solver.subsolver) 
          end
          
          set_objective!(stats, ft)
          unbounded = ft < fmin
          norm_∇fk = norm(∇fk)
          
          if ρk >= η2
            σk = max(σmin, γ3 * σk)
          else
            σk = γ1 * σk
          end
        else
          σk = γ2 * σk
        end
      end
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
    set_dual_residual!(stats, norm_∇fk)

    solver.σ = σk
    solver.subtol = subtol

    callback(nlp, solver, stats)

    norm_∇fk = stats.dual_feas
    σk = solver.σ
    subtol = solver.subtol
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([
        stats.iter, stats.objective, norm_∇fk, norm(s), σk, ρk, subiter, dir_stat, sub_stats
      ])
    end

    if stats.status == :user
      done = true
    else
      set_status!(
        stats,
        get_status(
          nlp,
          elapsed_time = stats.elapsed_time,
          optimal = optimal,
          unbounded = unbounded,
          max_eval = max_eval,
          iter = stats.iter,
          max_iter = max_iter,
          max_time = max_time,
        ),
      )
      done = stats.status != :unknown
    end
  end

  set_solution!(stats, x)
  return stats
end