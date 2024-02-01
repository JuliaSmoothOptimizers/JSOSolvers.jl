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
- `γ1 = T(1/2)`, `γ2 = T(2)`: regularization update parameters.
- `γ3 = T(1/2)` : momentum factor satβ update parameter in case of unsuccessful iteration.
- `αmax = 1/eps(T)`: step parameter for fomo algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1)` : target decay rate for the momentum.
- `θ1 = T(0.1)` : momentum contribution parameter for convergence condition #1. [(1-satβ)∇f(xk) + satβ mk.∇f(xk)] ≥ θ1||∇f(xk)||², with mk memory of past gradient and satβ ∈ [0,β].
- `θ2 = T(1e-5)` : momentum contribution parameter for convergence condition #2. ||∇f(xk)|| ≥ θ2||(1-satβ)∇f(xk) + satβ mk.∇f(xk)||, with mk memory of past gradient and satβ ∈ [0,β]. 
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
  d::V
end

function FomoSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  g = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  d = fill!(similar(nlp.meta.x0), 0)
  return FomoSolver{T, V}(x, g, c, m, d)
end

@doc (@doc FomoSolver) function fomo(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = FomoSolver(nlp)
  solver_specific = Dict(:avgsatβ => T(0.))
  stats = GenericExecutionStats(nlp;solver_specific=solver_specific)
  return solve!(solver, nlp, stats; kwargs...)
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
  η2 = T(0.95),
  γ1 = T(0.5),
  γ2 = T(2),
  γ3 = T(1/2),
  αmax = 1/eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β::T = T(0.9),
  θ1::T = T(1e-5),
  θ2::T = T(1e-5),
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
  d = solver.d
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

  d .= ∇fk
  norm_d = norm_∇fk
  satβ = T(0)
  ρk = T(0)
  avgsatβ = T(0.)
  siter = 0

  #μ = αk
  while !done
    λk = step_mult(αk,norm_d,backend)
    c .= x .- λk .* d
    ΔTk = dot(∇fk , d) * λk
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
      satβ *= γ3
      d .= ∇fk .* (T(1) - satβ) .+ m .* satβ
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= c
      if β!=0
        #μ = αk * (T(1) - β) + αk * β
        #m .= (αk/μ) .* ∇fk .* (T(1) - β) .+ m .* β
        m .= ∇fk .* (T(1) - β) .+ m .* β
      end
      #αk = μ
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      if β!= 0
        satβ = find_beta(m, ∇fk, norm_∇fk, β, θ1, θ2)
        d .= ∇fk .* (T(1) - satβ) .+ m .* satβ
        norm_d = norm(d)
      else
        d .= ∇fk
        norm_d = norm_∇fk
      end
      avgsatβ += satβ
      siter += 1
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
    
    αk == 0 && set_status!(stats,:exception)
    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  avgsatβ /= siter
  stats.solver_specific[:avgsatβ] = avgsatβ
  set_solution!(stats, x)
  return stats
end

"""
find_beta(m, ∇f, norm_∇f, β, θ1, θ2)

Compute satβ which saturates the contibution of the momentum term to the gradient.
satβ is computed such that m.∇f > θ * norm_∇f^2
""" 
function find_beta(m::V,∇f::V,norm_∇f::T, β::T, θ1::T, θ2::T) where {T,V}
  dotprod = dot(m,∇f)
  β1 = dotprod < norm_∇f^2 ? (1-θ1)*norm_∇f^2/(norm_∇f^2 - dotprod) : β
  β2 = (1-θ2)*norm_∇f/(θ2*norm(m .- ∇f))
  return min(β,min(β1,β2)) 
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