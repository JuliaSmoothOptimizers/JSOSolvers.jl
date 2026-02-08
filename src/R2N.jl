export R2N, R2NSolver, R2NParameterSet
export ShiftedLBFGSSolver, HSLDirectSolver

using LinearOperators, LinearAlgebra
using SparseArrays
using HSL

Julia

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
    Parameter(η1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(η2, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(γ1, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ3, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(δ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(σmin, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(non_mono_size, IntegerRange(1, typemax(Int))),
    Parameter(ls_c, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)), # c is typically (0, 1)
    Parameter(ls_increase, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)), # increase > 1
    Parameter(ls_decrease, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)), # decrease < 1
    Parameter(ls_min_alpha, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(ls_max_alpha, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
  )
end

abstract type AbstractShiftedLBFGSSolver end

mutable struct ShiftedLBFGSSolver <: AbstractShiftedLBFGSSolver #TODO Ask what I can do inside here
  # Shifted LBFGS-specific fields
end

abstract type AbstractMASolver end

"""
    HSLDirectSolver
Generic wrapper for HSL direct solvers (e.g., MA97, MA57).

# Fields
- `hsl_obj`: The HSL solver object (e.g., Ma97 or Ma57).
- `rows`, `cols`, `vals`: COO format for the Hessian + shift.
- `n`: Problem size.
- `nnzh`: Number of Hessian nonzeros.
- `work`: Workspace for solves (used for MA57).
"""
mutable struct HSLDirectSolver{T, S} <: AbstractMASolver
  hsl_obj::S
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector{T}
  n::Int
  nnzh::Int
  work::Vector{T} # workspace for solves # used for ma57 solver 
end

"""
    HSLDirectSolver(nlp, hsl_constructor)
Constructs an HSLDirectSolver for the given NLP model and HSL solver constructor (e.g., ma97_coord or ma57_coord).
"""
function HSLDirectSolver(nlp::AbstractNLPModel{T, V}, hsl_constructor) where {T, V}
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  total_nnz = nnzh + n

  rows = Vector{Int}(undef, total_nnz)
  cols = Vector{Int}(undef, total_nnz)
  vals = Vector{T}(undef, total_nnz)

  hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))
  hess_coord!(nlp, nlp.meta.x0, view(vals, 1:nnzh))

  @inbounds for i = 1:n
    rows[nnzh + i] = i
    cols[nnzh + i] = i
    vals[nnzh + i] = one(T)
  end

  hsl_obj = hsl_constructor(n, cols, rows, vals)
  if hsl_constructor == ma57_coord
    work = Vector{T}(undef, n * size(nlp.meta.x0, 2)) # size(b, 2)
  else
    work = Vector{T}(undef, 0) # No workspace needed for MA97
  end
  return HSLDirectSolver{T, typeof(hsl_obj)}(hsl_obj, rows, cols, vals, n, nnzh, work)
end

const npc_handler_allowed = [:gs, :sigma, :prev, :cp]

const R2N_allowed_subsolvers = [:cg, :cr, :minres, :minres_qlp, :shifted_lbfgs, :ma97, :ma57]

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
- `subsolver::Symbol = :cg`: the subsolver to solve the shifted system. 
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
  Op <: Union{AbstractLinearOperator{T}, AbstractMatrix{T}}, #TODO confirm with Prof. Orban
  ShiftedOp <: Union{ShiftedOperator{T, V, Op}, Nothing}, # for cg and cr solvers
  Sub <: Union{KrylovWorkspace{T, T, V}, ShiftedLBFGSSolver, HSLDirectSolver{T, S} where S},
  M <: AbstractNLPModel{T, V},
} <: AbstractOptimizationSolver
  x::V              # Current iterate x_k
  xt::V             # Trial iterate x_{k+1}
  gx::V             # Gradient ∇f(x)
  gn::V             # Gradient at new point (Quasi-Newton)
  Hs::V             # Storage for H*s products
  s::V              # Step direction
  scp::V            # Cauchy point step
  obj_vec::V        # History of objective values for non-monotone strategy
  H::Op             # Hessian operator
  A::ShiftedOp      # Shifted Operator (H + σI)
  r2_subsolver::Sub # The subproblem solver
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
  subsolver::Symbol = :cg,
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
  subsolver in R2N_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2N_allowed_subsolvers)")

  value(params.non_mono_size) >= 1 || error("non_mono_size must be greater than or equal to 1")

  !(subsolver == :shifted_lbfgs) ||
    (nlp isa LBFGSModel) ||
    error("Unsupported subsolver type, ShiftedLBFGSSolver can only be used by LBFGSModel")

  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  A = nothing

  local H, r2_subsolver

  if subsolver == :ma97
    LIBHSL_isfunctional() || error("HSL library is not functional")
    r2_subsolver = HSLDirectSolver(nlp, ma97_coord)
    H = spzeros(T, nvar, nvar)#TODO change this 
  elseif subsolver == :ma57
    LIBHSL_isfunctional() || error("HSL library is not functional")
    r2_subsolver = HSLDirectSolver(nlp, ma57_coord)
    H = spzeros(T, nvar, nvar)#TODO change this 
  else
    if subsolver == :shifted_lbfgs
      H = nlp.op
      r2_subsolver = ShiftedLBFGSSolver()
    else
      H = hess_op!(nlp, x, Hs)
      r2_subsolver = krylov_workspace(Val(subsolver), nvar, nvar, V)
      if subsolver in (:cg, :cr)
        A = ShiftedOperator(H)
      end
    end
  end

  Op = typeof(H)
  ShiftedOp = typeof(A)
  σ = zero(T)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  subtol = one(T)
  obj_vec = fill(typemin(T), non_mono_size)
  Sub = typeof(r2_subsolver)

  # Initialize LineModel pointing to x and s. We will redirect it later, but we initialize it here.
  h = LineModel(nlp, x, s)

  return R2NSolver{T, V, Op, ShiftedOp, Sub, typeof(nlp)}(
    x,
    xt,
    gx,
    gn,
    H,
    A,
    Hs,
    s,
    scp,
    obj_vec,
    r2_subsolver,
    subtol,
    σ,
    params,
    h,
  )
end

function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  reset!(solver.H)
  # If using Krylov subsolvers, update the shifted operator
  if solver.r2_subsolver isa CgWorkspace || solver.r2_subsolver isa CrWorkspace
    solver.A = ShiftedOperator(solver.H)
  end
  # No need to touch solver.h here; it is still valid for the current NLP
  return solver
end

function SolverCore.reset!(solver::R2NSolver{T}, nlp::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  reset!(solver.H)
  # If using Krylov subsolvers, update the shifted operator
  if solver.r2_subsolver isa CgWorkspace || solver.r2_subsolver isa CrWorkspace
    solver.A = ShiftedOperator(solver.H)
  end

  # We must create a new LineModel because the NLP has changed
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
  subsolver::Symbol = :cg,
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

  reset!(stats)
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
  ls_min_alpha = value(params.ls_min_alpha)
  ls_max_alpha = value(params.ls_max_alpha)

  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.meta.nvar
  x = solver.x .= x
  xt = solver.xt
  ∇fk = solver.gx # k-1
  ∇fn = solver.gn #current 
  s = solver.s
  scp = solver.scp
  H = solver.H
  A = solver.A
  Hs = solver.Hs
  σk = solver.σ

  subtol = solver.subtol
  subsolver_solved = false

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, ∇fk)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  ρk = zero(T)

  # Stopping criterion: 
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
  # cp_step_log = " "
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :f, :dual, :norm_s, :σ, :ρ, :sub_iter, :dir, :sub_status],
      [Int, Float64, Float64, Float64, Float64, Float64, Int, String, String],
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

  # subtol initialization for subsolver 
  subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
  solver.σ = σk
  solver.subtol = subtol
  r2_subsolver = solver.r2_subsolver

  if r2_subsolver isa ShiftedLBFGSSolver
    scp_flag = false # we don't need to do scp comparison for shifted lbfgs no matter what user says
  end

  callback(nlp, solver, stats)
  subtol = solver.subtol
  σk = solver.σ

  done = stats.status != :unknown

  ν_k = one(T) # used for scp calculation
  γ_k = zero(T)

  # Initialize variables to avoid scope warnings (although not strictly necessary if logic is sound)
  ft = f0

  while !done
    npcCount = 0
    fck_computed = false # Reset flag for optimization

    # Solving for step direction s_k
    ∇fk .*= -1
    if r2_subsolver isa CgWorkspace || r2_subsolver isa CrWorkspace
      # Update the shift in the operator
      solver.A.σ = σk
      solver.H = H
    end
    subsolver_solved, sub_stats, subiter, npcCount =
      subsolve!(r2_subsolver, solver, nlp, s, zero(T), n, subsolver_verbose)

    if !subsolver_solved && npcCount == 0
      @warn("Subsolver failed to solve the system")
      # TODO exit cleaning, update stats
      break
    end
    calc_scp_needed = false
    force_sigma_increase = false

    if r2_subsolver isa HSLDirectSolver
      num_neg, num_zero = get_inertia(r2_subsolver)
      coo_sym_prod!(r2_subsolver.rows, r2_subsolver.cols, r2_subsolver.vals, s, Hs)

      # Check Singularity (Zero Eigenvalues)          # assume Inconsistent could happen then increase the sigma
      if num_zero > 0
        force_sigma_increase = true #TODO Prof. Orban, for now we just increase sigma when we have zero eigenvalues
      end

      # Check Indefinite
      if !force_sigma_increase && num_neg > 0
        # curv_s = s' (H+σI) s
        curv_s = dot(s, Hs)

        if curv_s < 0
          npcCount = 1
          if npc_handler == :prev
            npc_handler = :gs  #Force the npc_handler to be gs and not :prev since we can not have that
          end
        else
          # Step has positive curvature, but matrix has negative eigs.
          #"compute scp and compare"
          calc_scp_needed = true
        end
      end
    end

    if !(r2_subsolver isa ShiftedLBFGSSolver) && npcCount >= 1  #npc case
      if npc_handler == :gs # Goldstein Line Search
        npcCount = 0

        if r2_subsolver isa HSLDirectSolver
          dir = s
        else
          dir = r2_subsolver.npc_dir
        end

        SolverTools.redirect!(solver.h, x, dir)

        f0_val = stats.objective
        # ∇fk is currently -∇f, so we negate it to get +∇f for the dot product
        slope = -dot(∇fk, dir)

        α, ft, nbk, nbG = armijo_goldstein(
          solver.h,
          f0_val,
          slope;
          t = one(T),       # Initial step
          τ₀ = ls_c,
          τ₁ = 1 - ls_c,
          γ₀ = ls_decrease, # Backtracking factor
          γ₁ = ls_increase, # Look-ahead factor
          bk_max = 100,     # Or add a param for this
          bG_max = 100,
          verbose = (verbose > 0),
        )
        @. s = α * dir
        fck_computed = true # Set flag to indicate ft is already computed from line search
      elseif npc_handler == :prev #Cr and cg will return the last iteration s
        npcCount = 0
        s .= r2_subsolver.x
      end
    end

    ∇fk .*= -1 # flip back to original +∇f
    # Compute the Cauchy step.
    if scp_flag == true || npc_handler == :cp || calc_scp_needed
      if r2_subsolver isa HSLDirectSolver
        coo_sym_prod!(r2_subsolver.rows, r2_subsolver.cols, r2_subsolver.vals, ∇fk, Hs)
      else
        mul!(Hs, H, ∇fk) # Use linear operator
      end

      curv = dot(∇fk, Hs)
      slope = σk * norm_∇fk^2 # slope= σ * ||∇f||^2 
      γ_k = (curv + slope) / norm_∇fk^2

      if γ_k > 0
        # cp_step_log = "α_k"
        ν_k = 2*(1-δ1) / (γ_k)
      else
        # we have to calcualte the scp, since we have encounter a negative curvature
        if H isa AbstractLinearOperator
          λmax, found_λ = opnorm(H) # This uses iterative methods (p=2)
        else
          λmax = norm(view(r2_subsolver.vals, 1:r2_subsolver.nnzh)) # f-norm of the H  #TODO double check if we need sigma
          found_λ = true # We assume the Inf-norm was found
        end
        # cp_step_log = "ν_k"
        ν_k = θ1 / (λmax + σk)
      end

      # Based on the flag, scp is calcualted
      mul!(scp, ∇fk, -ν_k)
      if npc_handler == :cp && npcCount >= 1
        npcCount = 0
        s .= scp
      elseif norm(s) > θ2 * norm(scp)
        s .= scp
      end
    end

    ∇fk .*= -1 # flip to -∇f 
    if force_sigma_increase || (npc_handler == :sigma && npcCount >= 1) # non-positive curvature case happen and the npc_handler is sigma
      step_accepted = false
      σk = max(σmin, γ2 * σk)
      solver.σ = σk
      npcCount = 0 # reset for next iteration
      ∇fk .*= -1
    else
      # Correctly compute curvature s' * B * s
      if solver.r2_subsolver isa HSLDirectSolver
        coo_sym_prod!(
          solver.r2_subsolver.rows,
          solver.r2_subsolver.cols,
          solver.r2_subsolver.vals,
          s,
          Hs,
        )
      else
        mul!(Hs, H, s) # Use linear operator
      end

      curv = dot(s, Hs)
      slope = dot(s, ∇fk) # = -∇fkᵀ s because we flipped the sign of ∇fk

      ΔTk = slope - curv / 2
      @. xt = x + s

      # OPTIMIZATION: Only calculate obj if Goldstein didn't already do it
      if !fck_computed
        ft = obj(nlp, xt)
      end

      if non_mono_size > 1  #non-monotone behaviour
        k = mod(stats.iter, non_mono_size) + 1
        solver.obj_vec[k] = stats.objective
        fck_max = maximum(solver.obj_vec)
        ρk = (fck_max - ft) / (fck_max - stats.objective + ΔTk) #TODO Prof. Orban check if this is correct the denominator   
      else
        # Avoid division by zero/negative. If ΔTk <= 0, the model is bad.
        ρk = (ΔTk > 10 * eps(T)) ? (stats.objective - ft) / ΔTk : -one(T)
        # ρk = (stats.objective - ft) / ΔTk
      end

      # Update regularization parameters and Acceptance of the new candidate
      step_accepted = ρk >= η1
      if step_accepted
        # update H implicitly
        x .= xt
        grad!(nlp, x, ∇fk)
        if isa(nlp, QuasiNewtonModel)
          ∇fn .-= ∇fk
          ∇fn .*= -1  # = ∇f(xₖ₊₁) - ∇f(xₖ)
          push!(nlp, s, ∇fn)
          ∇fn .= ∇fk
        end
        set_objective!(stats, ft)
        unbounded = ft < fmin
        norm_∇fk = norm(∇fk)
        if ρk >= η2
          σk = max(σmin, γ3 * σk)
        else # η1 ≤ ρk < η2
          σk = γ1 * σk
        end
        # we need to update H if we use Ma97 or ma57
        if solver.r2_subsolver isa HSLDirectSolver
          hess_coord!(nlp, x, view(solver.r2_subsolver.vals, 1:solver.r2_subsolver.nnzh))
        end
      else # η1 > ρk
        σk = γ2 * σk
        ∇fk .*= -1
      end
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
    set_dual_residual!(stats, norm_∇fk)

    solver.σ = σk
    solver.subtol = subtol

    callback(nlp, solver, stats)

    norm_∇fk = stats.dual_feas # if the user change it, they just change the stats.norm , they also have to change subtol
    σk = solver.σ
    subtol = solver.subtol
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([
        stats.iter,
        stats.objective,
        norm_∇fk,
        norm(s),
        σk,
        ρk,
        subiter,
        dir_stat,
        sub_stats,
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

# Dispatch for subsolvers KrylovWorkspace: cg and cr
function subsolve!(
  r2_subsolver::KrylovWorkspace{T, T, V},
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T, V}
  # Reset counters including npcCount (Bug Fix)
  r2_subsolver.stats.niter, r2_subsolver.stats.npcCount = 0, 0
  krylov_solve!(
    r2_subsolver,
    R2N.A, # Use the ShiftedOperator A
    R2N.gx,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.subtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver),
  r2_subsolver.stats.status,
  r2_subsolver.stats.niter,
  r2_subsolver.stats.npcCount
end

# Dispatch for MinresWorkspace and MinresQlpWorkspace
function subsolve!(
  r2_subsolver::Union{MinresWorkspace{T, V}, MinresQlpWorkspace{T, V}},
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T, V}
  # Reset counters including npcCount (Bug Fix)
  r2_subsolver.stats.niter, r2_subsolver.stats.npcCount = 0, 0
  krylov_solve!(
    r2_subsolver,
    R2N.H,
    R2N.gx,
    λ = R2N.σ,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.subtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver),
  r2_subsolver.stats.status,
  r2_subsolver.stats.niter,
  r2_subsolver.stats.npcCount
end

# Dispatch for ShiftedLBFGSSolver
function subsolve!(
  r2_subsolver::ShiftedLBFGSSolver,
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
)
  ∇f_neg = R2N.gx
  H = R2N.H
  σ = R2N.σ
  solve_shifted_system!(s, H, ∇f_neg, σ)
  return true, :first_order, 1, 0
end

# Dispatch for HSLDirectSolver
"""
    subsolve!(r2_subsolver::HSLDirectSolver, ...)
Solves the shifted system using the selected HSL direct solver (MA97 or MA57).
"""
# Wrapper for MA97
function get_inertia(solver::HSLDirectSolver{T, S}) where {T, S <: Ma97{T}}
  # MA97 provides num_neg directly. Rank is used to find num_zero.
  n = solver.n
  num_neg = solver.hsl_obj.info.num_neg
  # If matrix is full rank, num_zero is 0.
  num_zero = n - solver.hsl_obj.info.matrix_rank
  return num_neg, num_zero
end

# Wrapper for MA57
function get_inertia(solver::HSLDirectSolver{T, S}) where {T, S <: Ma57{T}}
  # MA57 uses different field names
  n = solver.n
  num_neg = solver.hsl_obj.info.num_negative_eigs
  num_zero = n - solver.hsl_obj.info.rank
  return num_neg, num_zero
end

# Fallback for other solvers (ShiftedLBFGS, Krylov)
# They don't provide direct inertia, so we return -1 (unknown)
get_inertia(solver) = (-1, -1)

"""
Internal helper for HSLDirectSolver: fallback if an unsupported HSL type is used.
"""
function _hsl_factor_and_solve!(solver::HSLDirectSolver{T, S}, g, s) where {T, S}
  error("Unsupported HSL solver type $(S)")
end

"""
Factorize and solve using MA97.
"""
# --- MA97 Implementation ---
function _hsl_factor_and_solve!(solver::HSLDirectSolver{T, S}, g, s) where {T, S <: Ma97{T}}
  ma97_factorize!(solver.hsl_obj)

  # Check for fatal errors only (flag < 0). 
  # Warnings (flag > 0) usually imply singularity, which we handle in the main loop.
  if solver.hsl_obj.info.flag < 0
    return false, :err, 0, 0
  end

  # Solve (MA97 handles singular systems by returning a solution)
  s .= g
  ma97_solve!(solver.hsl_obj, s)

  return true, :first_order, 1, 0
end

"""
Factorize and solve using MA57.
"""
function _hsl_factor_and_solve!(solver::HSLDirectSolver{T, S}, g, s) where {T, S <: Ma57{T}}
  ma57_factorize!(solver.hsl_obj)

  # MA57 returns flag=4 for singular matrices. This is NOT an error for us.
  # We only return false if it's a fatal error (flag < 0).
  # if solver.hsl_obj.info.flag < 0 #TODO we have flag in fortan but not on the Julia
  # return false, :err, 0, 0
  # end

  # Solve
  s .= g
  ma57_solve!(solver.hsl_obj, s, solver.work)

  return true, :first_order, 1, 0
end

"""
    subsolve!(r2_subsolver::HSLDirectSolver, ...)
Multiple-dispatch wrapper: updates the shifted diagonal then delegates to a
solver-specific `_hsl_factor_and_solve!` method (MA97 / MA57).
"""
function subsolve!(
  r2_subsolver::HSLDirectSolver{T, S},
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T, S}
  g = R2N.gx
  σ = R2N.σ
  @inbounds for i = 1:n
    r2_subsolver.vals[r2_subsolver.nnzh + i] = σ
  end
  return _hsl_factor_and_solve!(r2_subsolver, g, s)
end
