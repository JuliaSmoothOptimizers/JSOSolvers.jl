export fomo, FomoSolver, FoSolver, R2, tr_step, r2_step

abstract type AbstractFirstOrderSolver <: AbstractOptimizationSolver end

abstract type AbstractFomoMethod end
struct tr_step   <: AbstractFomoMethod end
struct r2_step   <: AbstractFomoMethod end

"""
    fomo(nlp; kwargs...)
    R2(nlp; kwargs...)

A First-Order with MOmentum (FOMO) model-based method for unconstrained optimization. Supports quadratic regularization and trust region steps.

For advanced usage, first define a `FomoSolver` or `FoSolver` to preallocate the memory used in the solver, and then call `solve!`:

    solver = FomoSolver(nlp)
    solve!(solver, nlp; kwargs...)

**Quadratic Regularization (R2)**: if the user do not want to use momentum (`β` = 0), it is recommended to use the memory-optimized `R2` method.
For advanced usage:

    solver = FoSolver(nlp)
    solve!(solver, nlp; kwargs...)
Extra keyword arguments `σmin` is accepted (`αmax` will be set to `1/σmin`).

# Arguments

- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 

- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `γ1 = T(1/2)`, `γ2 = T(2)`: regularization/trust region update parameters.
- `γ3 = T(1/2)` : momentum factor βmax update parameter in case of unsuccessful iteration.
- `αmax = 1/eps(T)`: maximum step parameter for fomo solver.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (-1 means unlimited).
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0.9) ∈ [0,1)` : target decay rate for the momentum.
- `θ1 = T(0.1)` : momentum contribution parameter for convergence condition #1. (1-βmax) * ∇f(xk) + βmax * dot(m,∇f(xk)) ≥ θ1 * ‖∇f(xk)‖², with m memory of past gradient and βmax ∈ [0,β].
- `θ2::T = T(eps(T)^(1/3))` : momentum contribution parameter for convergence condition #2. ‖∇f(xk)‖ ≥ θ2 * ‖(1-βmax) * ∇f(xk) + βmax * m‖, with m memory of past gradient and βmax ∈ [0,β]. 
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `step_backend = r2_step()`: step computation mode. Options are `r2_step()` for quadratic regulation step and `tr_step()` for first-order trust-region.

# Output

The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback

The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
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
## `R2`
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2(nlp)

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
mutable struct FomoSolver{T, V} <: AbstractFirstOrderSolver
  x::V
  g::V
  c::V
  m::V
  d::V
  p::V
  α::T
end

function FomoSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  d = fill!(similar(nlp.meta.x0), 0)
  p = similar(nlp.meta.x0)
  return FomoSolver{T, V}(x, g, c, m, d, p, T(0))
end

@doc (@doc FomoSolver) function fomo(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = FomoSolver(nlp)
  solver_specific = Dict(:avgβmax => T(0.))
  stats = GenericExecutionStats(nlp;solver_specific=solver_specific)
  return solve!(solver, nlp, stats; kwargs...)
end

function SolverCore.reset!(solver::FomoSolver{T}) where {T}
  fill!(solver.m,0)
  solver
end

SolverCore.reset!(solver::FomoSolver, ::AbstractNLPModel) = reset!(solver)

mutable struct FoSolver{T, V} <: AbstractFirstOrderSolver
  x::V
  g::V
  c::V
  α::T
end

function FoSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  return FoSolver{T, V}(x, g, c, T(0))
end

@doc (@doc FomoSolver) function R2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = FoSolver(nlp)
  stats = GenericExecutionStats(nlp)
  if haskey(kwargs,:σmin)
    return solve!(solver, nlp, stats; step_backend = r2_step(), αmax = 1/kwargs[:σmin], kwargs...)
  else
    return solve!(solver, nlp, stats; step_backend = r2_step(), kwargs...)
  end
end

function SolverCore.reset!(solver::FoSolver{T}) where {T}
  solver
end

SolverCore.reset!(solver::FoSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::AbstractFirstOrderSolver,
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1::T = T(eps(T)^(1 / 4)),
  η2::T = T(0.95),
  γ1::T = T(1/2),
  γ2::T = T(2),
  γ3::T = T(1/2),
  αmax::T = 1/eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β::T = T(0.9),
  θ1::T = T(0.1),
  θ2::T = T(eps(T)^(1/3)),
  verbose::Int = 0,
  step_backend = r2_step(),
  σmin = nothing # keep consistency with R2 interface. kwargs immutable, can't delete it in `R2`
) where {T, V}
  use_momentum = typeof(solver) <: FomoSolver
  unconstrained(nlp) || error("fomo should only be called on unconstrained problems.")
  
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
  set_objective!(stats, obj(nlp, x))

  
  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  solver.α = init_alpha(norm_∇fk,step_backend)
  
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    if !use_momentum
      @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
      @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk 1/solver.α
    else
      @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "α"
      @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.α
    end
    
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    if !use_momentum
      @info @sprintf "%5s  %9s  %7s  %7s  %7s " "iter" "f" "‖∇f‖" "σ" "ρk"
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk 1/solver.α NaN
    else
      @info @sprintf "%5s  %9s  %7s  %7s  %7s  %7s " "iter" "f" "‖∇f‖" "α" "ρk" "βmax"
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.α NaN 0
    end
  end

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
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
  siter = 0
  oneT = T(1)
  mdot∇f = T(0) # dot(momentum,∇fk)
  while !done
    λk = step_mult(solver.α,norm_d,step_backend)
    c .= x .- λk .* d
    step_underflow = x == c # step addition underfow on every dimensions, should happen before solver.α == 0
    ΔTk = ((oneT - βmax) * norm_∇fk^2 + βmax * mdot∇f) * λk # = dot(d,∇fk) * λk with momentum, ‖∇fk‖²λk without momentum
    fck = obj(nlp, c)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end
    ρk = (stats.objective - fck) / ΔTk
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
        mdot∇f = dot(momentum,∇fk)
      end
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      if use_momentum
        p .= momentum .- ∇fk
        βmax = find_beta(p , mdot∇f, norm_∇fk, β, θ1, θ2)
        d .= ∇fk .* (oneT - βmax) .+ momentum .* βmax
        norm_d = norm(d)
      end
      if use_momentum
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
      if !use_momentum
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk 1/solver.α ρk
      else
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.α ρk βmax
      end
    end

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )
    
    callback(nlp, solver, stats)

    step_underflow  && set_status!(stats,:small_step)
    solver.α == 0         && set_status!(stats,:exception) # :small_nlstep exception should happen before

    done = stats.status != :unknown
  end
  if use_momentum
    avgβmax /= siter
    stats.solver_specific[:avgβmax] = avgβmax
  end
  set_solution!(stats, x)
  return stats
end

"""
find_beta(m, md∇f, norm_∇f, β, θ1, θ2)

Compute βmax which saturates the contibution of the momentum term to the gradient.
`βmax` is computed such that the two gradient-related conditions are ensured: 
1. [(1-βmax) * ∇f(xk) + βmax * dot(m,∇f(xk))] ≥ θ1 * ‖∇f(xk)‖²
2. ‖∇f(xk)‖ ≥ θ2 * ‖(1-βmax) * ∇f(xk) + βmax * m‖
with `m` the momentum term and `mdot∇f = dot(m,∇f(xk))` 
""" 
function find_beta(p::V, mdot∇f::T, norm_∇f::T, β::T, θ1::T, θ2::T) where {T,V}
  n1 = norm_∇f^2 - mdot∇f
  n2 = norm(p)
  β1 = n1 > 0  ? (1-θ1)*norm_∇f^2/(n1)  : β
  β2 = n2 != 0 ? (1-θ2)*norm_∇f/(θ2*n2) : β
  return min(β,min(β1,β2))
end

"""
  init_alpha(norm_∇fk::T, ::r2_step)
  init_alpha(norm_∇fk::T, ::tr_step)

Initialize α step size parameter. Ensure first step is the same for quadratic regularization and trust region methods.
"""
function init_alpha(norm_∇fk::T, ::r2_step) where{T}
  1/2^round(log2(norm_∇fk + 1))
end

function init_alpha(norm_∇fk::T, ::tr_step) where{T}
  norm_∇fk/2^round(log2(norm_∇fk + 1))
end

"""
  step_mult(α::T, norm_∇fk::T, ::r2_step)
  step_mult(α::T, norm_∇fk::T, ::tr_step)

Compute step size multiplier: `α` for quadratic regularization(`::r2` and `::R2og`) and `α/norm_∇fk` for trust region (`::tr`).
"""
function step_mult(α::T, norm_∇fk::T, ::r2_step) where{T}
  α
end

function step_mult(α::T, norm_∇fk::T, ::tr_step) where{T}
  α/norm_∇fk
end