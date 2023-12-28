export fomo, FomoSolver, tr, qr

abstract type AbstractFomoMethod end

struct tr <: AbstractFomoMethod end
struct qr <: AbstractFomoMethod end

"""
    fomo(nlp; kwargs...)

A First-Order with MOmentum (FOMO) model-based method for unconstrained optimization. Supports quadratic regularization and trust region methods.

For advanced usage, first define a `FomoSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = FomoSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.2)`: step acceptance parameters.
- `κg = T(0.8)` : maximum contribution of momentum term to the gradient, ||∇f-g||≤κg||g|| with g = (1-β)∇f + β m, with m memory of past gradients. Must satisfy 0 < κg < 1 - η2.
- `γ1 = T(0.8)`, `γ2 = T(1.2)`: regularization update parameters.
- `αmax = 1/eps(T)`: step parameter for fomo algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1)` : constant in the momentum term.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `backend = qr()`: model-based method employed. Options are `qr()` for quadratic regulation and `tr()` for trust-region

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
mutable struct FomoSolver{T, V} <: AbstractOptimizationSolver
  x::V
  g::V
  c::V
  m::V
end

function FomoSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  return FomoSolver{T, V}(x, g, c, m)
end

@doc (@doc FomoSolver) function fomo(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = FomoSolver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::FomoSolver{T}) where {T}
  fill!(solver.m,0)
  solver
end
SolverCore.reset!(solver::FomoSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::FomoSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.2),
  κg = T(0.8),
  γ1 = T(0.5),
  γ2 = T(2),
  αmax = 1/eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β::T = T(0.9),
  verbose::Int = 0,
  backend = qr()
) where {T, V}
  unconstrained(nlp) || error("fomo should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.g
  c = solver.c
  m = solver.m

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  αk = init_alpha(norm_∇fk,backend)
  
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "α"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk αk
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s  %7s" "iter" "f" "‖∇f‖" "α" "staβ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk αk 0
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

  satβ = T(0)
  while !done
    λk = step_mult(αk,norm_∇fk,backend)
    if β == 0
      c .= x .- λk .* (∇fk)
    else
      c .= x .- λk .* (∇fk .* (T(1) - satβ) .+ m .* satβ)
    end
    ΔTk = norm_∇fk^2 * λk
    fck = obj(nlp, c)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end
    
    ρk = (stats.objective - fck) / ΔTk
    
    # Update regularization parameters
    if ρk >= η2
      αk = min(αmax, γ2 * αk)
    elseif ρk < η1
      αk = αk * γ1
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= c
      if β!=0
        m .= ∇fk .* (T(1) - β) .+ m .* β
      end
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      if β!=0
        satβ = find_beta(β, κg, m, ∇fk)
      end
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk αk satβ
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
  end

  set_solution!(stats, x)
  return stats
end

"""
  find_beta(β,κg,d,∇f;tol=0.01)

Compute satβ which saturates the contibution of the momentum term to the gradient.
Use bisection method to solve satβ * ||∇f .- d|| = κg * ||(1-satβ) .* ∇f + satβ .* d|| where d is the momentum term.
""" 
function find_beta(β::T,κg::T,d::V,∇f::V;tol=0.01) where {T,V}
  if β * norm( ∇f .- d) - κg * norm((1-β) .* ∇f + β .* d) <= 0.
    return β
  end
  a = T(0)
  b = β 
  while b-a > tol
    β = (b+a) / 2
    if β * norm( ∇f .- d) - κg * norm((1-β) .* ∇f + β .* d) <= 0     
      a = β
    else
      b = β
    end
  end
  return a
end

"""
  init_alpha(norm_∇fk::T, ::qr)
  init_alpha(norm_∇fk::T, ::tr)

Initialize α step size parameter. Ensure first step is the same for quadratic regularization and trust region methods.
"""
function init_alpha(norm_∇fk::T, ::qr) where{T}
  1/2^round(log2(norm_∇fk + 1))
end

function init_alpha(norm_∇fk::T, ::tr) where{T}
  norm_∇fk/2^round(log2(norm_∇fk + 1))
end

"""
  step_mult(αk::T, norm_∇fk::T, ::qr)
  step_mult(αk::T, norm_∇fk::T, ::tr)

Compute step size multiplier: `αk` for quadratic regularization(`::qr`) and `αk/norm_∇fk` for trust region (`::tr`).
"""
function step_mult(αk::T, norm_∇fk::T, ::qr) where{T}
  αk
end

function step_mult(αk::T, norm_∇fk::T, ::tr) where{T}
  αk/norm_∇fk
end