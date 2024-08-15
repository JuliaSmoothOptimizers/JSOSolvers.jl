export fomo, FomoSolver, FOMOParameterSet, FoSolver, fo, R2, TR
export tr_step, r2_step

abstract type AbstractFirstOrderSolver <: AbstractOptimizationSolver end

abstract type AbstractFOMethod end
struct tr_step <: AbstractFOMethod end
struct r2_step <: AbstractFOMethod end

# Default algorithm parameter values
const FOMO_η1 =
  DefaultParameter((nlp::AbstractNLPModel) -> eps(eltype(nlp.meta.x0))^(1 // 4), "eps(T)^(1 // 4)")
const FOMO_η2 =
  DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(95 // 100), "T(95/100)")
const FOMO_γ1 = DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(1 // 2), "T(1/2)")
const FOMO_γ2 = DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(2), "T(2)")
const FOMO_γ3 = DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(1 // 2), "T(1/2)")
const FOMO_αmax =
  DefaultParameter((nlp::AbstractNLPModel) -> 1 / eps(eltype(nlp.meta.x0)), "1/eps(T)")
const FOMO_β = DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(9 // 10), "T(9/10)")
const FOMO_θ1 = DefaultParameter((nlp::AbstractNLPModel) -> eltype(nlp.meta.x0)(1 // 10), "T(1/10)")
const FOMO_θ2 =
  DefaultParameter((nlp::AbstractNLPModel) -> eps(eltype(nlp.meta.x0))^(1 // 3), "eps(T)^(1/3)")
const FOMO_M = DefaultParameter(1)
const FOMO_step_backend = DefaultParameter(nlp -> r2_step(), "r2_step()")

"""
    FOMOParameterSet{T} <: AbstractParameterSet

This structure designed for `fomo` regroups the following parameters:
  - `η1`, `η2`: step acceptance parameters.
  - `γ1`, `γ2`: regularization update parameters.
  - `γ3` : momentum factor βmax update parameter in case of unsuccessful iteration.
  - `αmax`: maximum step parameter for fomo algorithm.
  - `β ∈ [0,1)`: target decay rate for the momentum.
  - `θ1`: momentum contribution parameter for convergence condition (1).
  - `θ2`: momentum contribution parameter for convergence condition (2). 
  - `M` : requires objective decrease over the `M` last iterates (nonmonotone context). `M=1` implies monotone behaviour. 
  - `step_backend`: step computation mode. Options are `r2_step()` for quadratic regulation step and `tr_step()` for first-order trust-region.

An additional constructor is

    FOMOParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `η1::T = $(FOMO_η1)`
  - `η2::T = $(FOMO_η2)`
  - `γ1::T = $(FOMO_γ1)`
  - `γ2::T = $(FOMO_γ2)`
  - `γ3::T = $(FOMO_γ3)`
  - `αmax::T = $(FOMO_αmax)`
  - `β = $(FOMO_β) ∈ [0,1)`
  - `θ1 = $(FOMO_θ1)`
  - `θ2 = $(FOMO_θ2)`
  - `M = $(FOMO_M)`
  - `step_backend = $(FOMO_step_backend)
"""
struct FOMOParameterSet{T} <: AbstractParameterSet
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  γ3::Parameter{T, RealInterval{T}}
  αmax::Parameter{T, RealInterval{T}}
  β::Parameter{T, RealInterval{T}}
  θ1::Parameter{T, RealInterval{T}}
  θ2::Parameter{T, RealInterval{T}}
  M::Parameter{Int, IntegerRange{Int}}
  step_backend::Parameter{Union{r2_step, tr_step}, CategoricalSet{Union{r2_step, tr_step}}}
end

# add a default constructor
function FOMOParameterSet(
  nlp::AbstractNLPModel{T};
  η1::T = get(FOMO_η1, nlp),
  η2::T = get(FOMO_η2, nlp),
  γ1::T = get(FOMO_γ1, nlp),
  γ2::T = get(FOMO_γ2, nlp),
  γ3::T = get(FOMO_γ3, nlp),
  αmax::T = get(FOMO_αmax, nlp),
  β::T = get(FOMO_β, nlp),
  θ1::T = get(FOMO_θ1, nlp),
  θ2::T = get(FOMO_θ2, nlp),
  M::Int = get(FOMO_M, nlp),
  step_backend::AbstractFOMethod = get(FOMO_step_backend, nlp),
) where {T}
  @assert η1 <= η2
  FOMOParameterSet(
    Parameter(η1, RealInterval(T(0), T(1), lower_open = true, upper_open = true)),
    Parameter(η2, RealInterval(T(0), T(1), lower_open = true, upper_open = true)),
    Parameter(γ1, RealInterval(T(0), T(1), lower_open = true, upper_open = true)),
    Parameter(γ2, RealInterval(T(1), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ3, RealInterval(T(0), T(1))),
    Parameter(αmax, RealInterval(T(1), T(Inf), upper_open = true)),
    Parameter(β, RealInterval(T(0), T(1), upper_open = true)),
    Parameter(θ1, RealInterval(T(0), T(1))),
    Parameter(θ2, RealInterval(T(0), T(1), upper_open = true)),
    Parameter(M, IntegerRange(Int(1), typemax(Int))),
    Parameter(step_backend, CategoricalSet{Union{tr_step, r2_step}}([r2_step(); tr_step()])),
  )
end

"""
    fomo(nlp; kwargs...)

A First-Order with MOmentum (FOMO) model-based method for unconstrained optimization. Supports quadratic regularization and trust region method with linear model.

# Algorithm description

The step is computed along
d = - (1-βmax) .* ∇f(xk) - βmax .* mk
with mk the memory of past gradients (initialized at 0), and updated at each successful iteration as
mk .= ∇f(xk) .* (1 - βmax) .+ mk .* βmax
and βmax ∈ [0,β] chosen as to ensure d is gradient-related, i.e., the following 2 conditions are satisfied:
(1-βmax) .* ∇f(xk) + βmax .* ∇f(xk)ᵀmk ≥ θ1 * ‖∇f(xk)‖² (1)
‖∇f(xk)‖ ≥ θ2 * ‖(1-βmax) *. ∇f(xk) + βmax .* mk‖       (2)
In the nonmonotone case, (1) rewrites
(1-βmax) .* ∇f(xk) + βmax .* ∇f(xk)ᵀmk + (fm - fk)/μk ≥ θ1 * ‖∇f(xk)‖²,
with fm the largest objective value over the last M successful iterations, and fk = f(xk).

# Advanced usage

For advanced usage, first define a `FomoSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = FomoSolver(nlp)
    solve!(solver, nlp; kwargs...)

**No momentum**: if the user does not whish to use momentum (`β` = 0), it is recommended to use the memory-optimized `fo` method.
    
# Arguments

- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 

- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = $(FOMO_η1)`, `η2 = $(FOMO_η2)`: step acceptance parameters.
- `γ1 = $(FOMO_γ1)`, `γ2 = $(FOMO_γ2)`: regularization update parameters.
- `γ3 = $(FOMO_γ3)` : momentum factor βmax update parameter in case of unsuccessful iteration.
- `αmax = $(FOMO_αmax)`: maximum step parameter for fomo algorithm.
- `max_eval::Int = -1`: maximum number of objective evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = $(FOMO_β) ∈ [0,1)`: target decay rate for the momentum.
- `θ1 = $(FOMO_θ1)`: momentum contribution parameter for convergence condition (1).
- `θ2 = $(FOMO_θ2)`: momentum contribution parameter for convergence condition (2). 
- `M = $(FOMO_M)` : requires objective decrease over the `M` last iterates (nonmonotone context). `M=1` implies monotone behaviour. 
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `step_backend = $(FOMO_step_backend)`: step computation mode. Options are `r2_step()` for quadratic regulation step and `tr_step()` for first-order trust-region.

# Output

The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback

$(Callback_docstring)

The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user || stats.stats = :unknown` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
    - `stats.dual_feas`: norm of current gradient;
    - `stats.iter`: current iteration counter;
    - `stats.objective`: current objective function value;
    - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
    - `stats.elapsed_time`: elapsed time in seconds.

# Examples

## `fomo`

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = fomo(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = FomoSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct FomoSolver{T, V} <: AbstractFirstOrderSolver
  x::V
  g::V
  c::V
  m::V
  d::V
  p::V
  o::V
  α::T
  params::FOMOParameterSet{T}
end

function FomoSolver(nlp::AbstractNLPModel{T, V}; M::Int = get(FOMO_M, nlp), kwargs...) where {T, V}
  params = FOMOParameterSet(nlp; M = M, kwargs...)
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  d = fill!(similar(nlp.meta.x0), 0)
  p = similar(nlp.meta.x0)
  o = fill!(Vector{T}(undef, M), -Inf)
  return FomoSolver{T, V}(x, g, c, m, d, p, o, T(0), params)
end

@doc (@doc FomoSolver) function fomo(
  nlp::AbstractNLPModel{T, V};
  η1::T = get(FOMO_η1, nlp),
  η2::T = get(FOMO_η2, nlp),
  γ1::T = get(FOMO_γ1, nlp),
  γ2::T = get(FOMO_γ2, nlp),
  γ3::T = get(FOMO_γ3, nlp),
  αmax::T = get(FOMO_αmax, nlp),
  β::T = get(FOMO_β, nlp),
  θ1::T = get(FOMO_θ1, nlp),
  θ2::T = get(FOMO_θ2, nlp),
  M::Int = get(FOMO_M, nlp),
  step_backend::AbstractFOMethod = get(FOMO_step_backend, nlp),
  kwargs...,
) where {T, V}
  solver = FomoSolver(
    nlp;
    η1 = η1,
    η2 = η2,
    γ1 = γ1,
    γ2 = γ2,
    γ3 = γ3,
    αmax = αmax,
    β = β,
    θ1 = θ1,
    θ2 = θ2,
    M = M,
    step_backend = step_backend,
  )
  solver_specific = Dict(:avgβmax => T(0.0))
  stats = GenericExecutionStats(nlp; solver_specific = solver_specific)
  return solve!(solver, nlp, stats; kwargs...)
end

function SolverCore.reset!(solver::FomoSolver{T}) where {T}
  fill!(solver.m, 0)
  fill!(solver.o, -Inf)
  solver
end

SolverCore.reset!(solver::FomoSolver, ::AbstractNLPModel) = reset!(solver)

"""
    fo(nlp; kwargs...)
    R2(nlp; kwargs...)
    TR(nlp; kwargs...)

A First-Order (FO) model-based method for unconstrained optimization. Supports quadratic regularization and trust region method with linear model.

For advanced usage, first define a `FomoSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = FoSolver(nlp)
    solve!(solver, nlp; kwargs...)

`R2` and `TR` runs `fo` with the dedicated `step_backend` keyword argument.

# Arguments

- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 

- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = $(FOMO_η1)`: algorithm parameter, see [`FOMOParameterSet`](@ref).
- `η2 = $(FOMO_η2)`: algorithm parameter, see [`FOMOParameterSet`](@ref).
- `γ1 = $(FOMO_γ1)`: algorithm parameter, see [`FOMOParameterSet`](@ref).
- `γ2 = $(FOMO_γ2)`: algorithm parameter, see [`FOMOParameterSet`](@ref).
- `αmax = $(FOMO_αmax)`: algorithm parameter, see [`FOMOParameterSet`](@ref).
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `M = $(FOMO_M)` : algorithm parameter, see [`FOMOParameterSet`](@ref).
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `step_backend = $(FOMO_step_backend)`: algorithm parameter, see [`FOMOParameterSet`](@ref).

# Output

The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback

$(Callback_docstring)

# Examples

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = fo(nlp) # run with step_backend = r2_step(), equivalent to R2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = FoSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct FoSolver{T, V} <: AbstractFirstOrderSolver
  x::V
  g::V
  c::V
  o::V
  α::T
  params::FOMOParameterSet{T}
end

function FoSolver(nlp::AbstractNLPModel{T, V}; M::Int = get(FOMO_M, nlp), kwargs...) where {T, V}
  params = FOMOParameterSet(nlp; M = M, kwargs...)
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  o = fill!(Vector{T}(undef, M), -Inf)
  return FoSolver{T, V}(x, g, c, o, T(0), params)
end

"""
    `R2Solver` is deprecated, please check the documentation of `R2`.
"""
mutable struct R2Solver{T, V} <: AbstractOptimizationSolver end

Base.@deprecate R2Solver(nlp::AbstractNLPModel; kwargs...) FoSolver(
  nlp::AbstractNLPModel;
  M = FOMO_M,
  kwargs...,
)

@doc (@doc FoSolver) function fo(
  nlp::AbstractNLPModel{T, V};
  η1::T = get(FOMO_η1, nlp),
  η2::T = get(FOMO_η2, nlp),
  γ1::T = get(FOMO_γ1, nlp),
  γ2::T = get(FOMO_γ2, nlp),
  γ3::T = get(FOMO_γ3, nlp),
  αmax::T = get(FOMO_αmax, nlp),
  β::T = get(FOMO_β, nlp),
  θ1::T = get(FOMO_θ1, nlp),
  θ2::T = get(FOMO_θ2, nlp),
  M::Int = get(FOMO_M, nlp),
  step_backend::AbstractFOMethod = get(FOMO_step_backend, nlp),
  kwargs...,
) where {T, V}
  solver = FoSolver(
    nlp;
    η1 = η1,
    η2 = η2,
    γ1 = γ1,
    γ2 = γ2,
    γ3 = γ3,
    αmax = αmax,
    β = β,
    θ1 = θ1,
    θ2 = θ2,
    M = M,
    step_backend = step_backend,
  )
  stats = GenericExecutionStats(nlp)
  return solve!(solver, nlp, stats; kwargs...)
end

@doc (@doc FoSolver) function R2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  fo(nlp; step_backend = r2_step(), kwargs...)
end

@doc (@doc FoSolver) function TR(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  fo(nlp; step_backend = tr_step(), kwargs...)
end

function SolverCore.reset!(solver::FoSolver{T}) where {T}
  fill!(solver.o, -Inf)
  solver
end

SolverCore.reset!(solver::FoSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::Union{FoSolver, FomoSolver},
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
) where {T, V}
  unconstrained(nlp) || error("fomo should only be called on unconstrained problems.")

  # parameters
  η1 = value(solver.params.η1)
  η2 = value(solver.params.η2)
  γ1 = value(solver.params.γ1)
  γ2 = value(solver.params.γ2)
  γ3 = value(solver.params.γ3)
  αmax = value(solver.params.αmax)
  β = value(solver.params.β)
  θ1 = value(solver.params.θ1)
  θ2 = value(solver.params.θ2)
  M = value(solver.params.M)
  step_backend = value(solver.params.step_backend)

  use_momentum = typeof(solver) <: FomoSolver
  is_r2 = typeof(step_backend) <: r2_step

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.g
  c = solver.c
  momentum = use_momentum ? solver.m : nothing # not used if no momentum
  d = use_momentum ? solver.d : solver.g # g = d if no momentum
  p = use_momentum ? solver.p : nothing # not used if no momentum
  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)
  obj_mem = solver.o
  M = length(obj_mem)
  mem_ind = 0
  obj_mem[mem_ind + 1] = stats.objective
  max_obj_mem = stats.objective

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  solver.α = init_alpha(norm_∇fk, step_backend)

  # Stopping criterion: 
  fmin = min(-one(T), f0) / eps(T)
  unbounded = f0 < fmin

  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  step_param_name = is_r2 ? "σ" : "Δ"
  if optimal
    @info("Optimal point found at initial point")
    if is_r2
      @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name
      @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk 1 / solver.α
    else
      @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name
      @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.α
    end
  else
    if verbose > 0 && mod(stats.iter, verbose) == 0
      step_param = is_r2 ? 1 / solver.α : solver.α
      if !use_momentum
        @info @sprintf "%5s  %9s  %7s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name "ρk"
        infoline =
          @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk step_param ' '
      else
        @info @sprintf "%5s  %9s  %7s  %7s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name "ρk" "βmax"
        infoline =
          @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk step_param ' ' 0
      end
    end
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

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  d .= ∇fk
  norm_d = norm_∇fk
  βmax = T(0)
  ρk = T(0)
  avgβmax = T(0)
  siter::Int = 0
  oneT = T(1)
  mdot∇f = T(0) # dot(momentum,∇fk)
  while !done
    μk = step_mult(solver.α, norm_d, step_backend)
    c .= x .- μk .* d
    step_underflow = x == c # step addition underfow on every dimensions, should happen before solver.α == 0
    ΔTk = ((oneT - βmax) * norm_∇fk^2 + βmax * mdot∇f) * μk # = dot(d,∇fk) * μk with momentum, ‖∇fk‖²μk without momentum
    fck = obj(nlp, c)
    unbounded = fck < fmin
    ρk = (max_obj_mem - fck) / (max_obj_mem - stats.objective + ΔTk)
    # Update regularization parameters
    if ρk >= η2
      solver.α = min(αmax, γ2 * solver.α)
    elseif ρk < η1
      solver.α = solver.α * γ1
      if use_momentum
        βmax *= γ3
        d .= ∇fk .* (oneT - βmax) .+ momentum .* βmax
      end
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= c
      if use_momentum
        momentum .= ∇fk .* (oneT - β) .+ momentum .* β
      end
      set_objective!(stats, fck)
      mem_ind = (mem_ind + 1) % M
      obj_mem[mem_ind + 1] = stats.objective
      max_obj_mem = maximum(obj_mem)

      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      if use_momentum
        mdot∇f = dot(momentum, ∇fk)
        p .= momentum .- ∇fk
        βmax = find_beta(p, mdot∇f, norm_∇fk, μk, stats.objective, max_obj_mem, β, θ1, θ2)
        d .= ∇fk .* (oneT - βmax) .+ momentum .* βmax
        norm_d = norm(d)
        avgβmax += βmax
        siter += 1
      end
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      step_param = is_r2 ? 1 / solver.α : solver.α
      if !use_momentum
        infoline =
          @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk step_param ρk
      else
        infoline =
          @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk step_param ρk βmax
      end
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

    callback(nlp, solver, stats)

    step_underflow && set_status!(stats, :small_step)
    solver.α == 0 && set_status!(stats, :exception) # :small_nlstep exception should happen before

    done = stats.status != :unknown
  end
  if use_momentum
    avgβmax /= siter
    set_solver_specific!(stats, :avgβmax, avgβmax)
  end
  set_solution!(stats, x)
  return stats
end

"""
    find_beta(m, mdot∇f, norm_∇f, μk, fk, max_obj_mem, β, θ1, θ2)

Compute βmax which saturates the contribution of the momentum term to the gradient.
`βmax` is computed such that the two gradient-related conditions (first one is relaxed in the nonmonotone case) are ensured: 
1. (1-βmax) * ‖∇f(xk)‖² + βmax * ∇f(xk)ᵀm + (max_obj_mem - fk)/μk ≥ θ1 * ‖∇f(xk)‖²
2. ‖∇f(xk)‖ ≥ θ2 * ‖(1-βmax) * ∇f(xk) .+ βmax .* m‖
with `m` the momentum term and `mdot∇f = ∇f(xk)ᵀm`, `fk` the model at s=0, `max_obj_mem` the largest objective value over the last M successful iterations.
"""
function find_beta(
  p::V,
  mdot∇f::T,
  norm_∇f::T,
  μk::T,
  fk::T,
  max_obj_mem::T,
  β::T,
  θ1::T,
  θ2::T,
) where {T, V}
  n1 = norm_∇f^2 - mdot∇f
  n2 = norm(p)
  β1 = n1 > 0 ? ((1 - θ1) * norm_∇f^2 - (fk - max_obj_mem) / μk) / n1 : β
  β2 = n2 != 0 ? (1 - θ2) * norm_∇f / n2 : β
  return min(β, min(β1, β2))
end

"""
    init_alpha(norm_∇fk::T, ::r2_step)
    init_alpha(norm_∇fk::T, ::tr_step)

Initialize `α` step size parameter.
Ensure first step is the same for quadratic regularization and trust region methods.
"""
function init_alpha(norm_∇fk::T, ::r2_step) where {T}
  1 / 2^round(log2(norm_∇fk + 1))
end

function init_alpha(norm_∇fk::T, ::tr_step) where {T}
  norm_∇fk / 2^round(log2(norm_∇fk + 1))
end

"""
    step_mult(α::T, norm_∇fk::T, ::r2_step)
    step_mult(α::T, norm_∇fk::T, ::tr_step)

Compute step size multiplier: `α` for quadratic regularization(`::r2` and `::R2og`) and `α/norm_∇fk` for trust region (`::tr`).
"""
function step_mult(α::T, norm_∇fk::T, ::r2_step) where {T}
  α
end

function step_mult(α::T, norm_∇fk::T, ::tr_step) where {T}
  α / norm_∇fk
end
