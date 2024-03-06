export tadam, TadamSolver

"""
    tadam(nlp; kwargs...)

Trust-region embeded ADAM (TADAM) algorithm for unconstrained optimization. This is an adaptation of ADAM which enforces convergence in the non-convexe case.

# Minimal algorithm description 
The step sk at iteration k is computed as:
sk = argmin -̂dkᵀs + 0.5 sᵀ diag(sqrt.(̂vk) + e) s      (1)
      s
s.t.   ||s||∞ <= Δk
where:
1. Δk is the trust-region radius
2. ̂vk is the biased-corrected second raw moment estimate
3. ̂dk = dk/(1-β1ᵏ) the biased-corrected restricted momentum direction
4. -dk = (1-β1max) .* ∇fk + β1max .* mk,  is restricted momentum direction, with mk the memory of past gradient and β1max computed such that d is gradient-related (necessary to ensure convergence).
5. β1max is computed so that dk is gradient related, i.e., the following 2 conditions are satisfied:
-̂dᵀ∇f(xk) ≥ θ1 * ‖∇f(xk)‖²  (2)
‖∇f(xk)‖ ≥ θ2 * ‖̂d‖         (3)

Note that the solution to (1) without the trust region constraint is the ADAM step if β1max = β1 (no momentum contribution restriction).

# Advanced usage
For advanced usage, first define a `TadamSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TadamSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `γ1 = T(1/2)`, `γ2 = T(2)`: regularization update parameters.
- `γ3 = T(1/2)` : momentum contribution decrease factor, applied if iteration is unsuccessful
- `Δmax = 1/eps(T)`: step parameter for tadam algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β1 = T(0.9) ∈ [0,1)` : constant in the momentum term.
- `β2 = T(0.999) ∈ [0,1)` : constant in the RMSProp term.
- `e = T(1e-8)` : RMSProp epsilon
- `θ1 = T(0.1)` : momentum contribution parameter for convergence condition (2).
- `θ2 = T(eps(T)^(1/3))` : momentum contribution parameter for convergence condition (3). 
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
stats = tadam(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = TadamSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct TadamSolver{T, V} <: AbstractOptimizationSolver
  x::V
  ∇f::V
  c::V
  m::V
  d::V
  v::V
  s::V
  p::V
  Δ::T
end

function TadamSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  ∇f = similar(nlp.meta.x0)
  c = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), 0)
  d = similar(nlp.meta.x0)
  v = fill!(similar(nlp.meta.x0), 0)
  s = similar(nlp.meta.x0)
  p = similar(nlp.meta.x0)
  return TadamSolver{T, V}(x, ∇f, c, m, d, v, s, p, T(0))
end

@doc (@doc TadamSolver) function tadam(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = TadamSolver(nlp)
  solver_specific = Dict(:avgβ1max => T(0.0))
  stats = GenericExecutionStats(nlp; solver_specific = solver_specific)
  return solve!(solver, nlp, stats; kwargs...)
end

function SolverCore.reset!(solver::TadamSolver{T}) where {T}
  fill!(solver.m,0)
  fill!(solver.v,0)
  solver
end
SolverCore.reset!(solver::TadamSolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::TadamSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  γ1 = T(1/2),
  γ2 = T(2),
  γ3 = T(1/2),
  Δmax = 1/eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β1::T = T(0.9),
  β2::T = T(0.99),
  e::T = T(1e-8),
  θ1::T = T(0.1),
  θ2::T = T(eps(T)^(1 / 3)),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("tadam should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.∇f
  c = solver.c
  momentum = solver.m
  d̂ = solver.d
  bc_second_momentum = solver.v # biased corrected raw second order momentum
  s = solver.s
  p = solver.p
  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  solver.Δ = norm_∇fk/2^round(log2(norm_∇fk + 1))
  
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "Δ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.Δ
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s  %7s" "iter" "f" "‖∇f‖" "Δ" "β1max"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.Δ ' '
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.Δ ' '
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
  
  
  d̂  .= - ∇fk # biased corrected
  bc_second_momentum .= ∇fk.^2 # biased corrected
  β1max = T(0)
  ρk = T(0)
  avgβ1max = T(0)
  siter = 0
  oneT = T(1)
  mdot∇f = T(0) # dot(momentum,∇fk)
  siter = 1 # nb of successful iterations
  while !done
    solve_tadam_subproblem!(s, d̂ ,bc_second_momentum , solver.Δ, e)
    c .= x .+ s
    step_underflow = x == c # step addition underfow on every dimensions, should happen before solver.α == 0
    ΔTk = dot(d̂, s) - T(0.5)*dot(s.^2, sqrt.(bc_second_momentum) .+ e)
    fck = obj(nlp, c)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end
    ρk = (stats.objective - fck) / ΔTk
    if ρk >= η2
      solver.Δ = min(Δmax, γ2 * solver.Δ)
    elseif ρk < η1
      solver.Δ = solver.Δ * γ1
      β1max *= γ3
      d̂ .= - (∇fk .* (oneT - β1max) .+ momentum .* β1max) ./ (oneT - β1^siter)
    end
    # Acceptance of the new candidate
    if ρk >= η1
      siter += 1
      x .= c
      set_objective!(stats, fck)
      momentum .= ∇fk .* (oneT - β1) .+ momentum .* β1
      bc_second_momentum .= (∇fk.^2 .* (oneT - β2) .+ bc_second_momentum .*β2 .*(oneT - β2^(siter-1)) ) ./ (oneT - β2^siter) # possibly unstable but avoid allocating two vectors for bias corrected and biased raw second order momentum
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      mdot∇f = dot(momentum, ∇fk)
      p .= momentum .- ∇fk
      β1max = find_beta(p, mdot∇f, norm_∇fk, β1, θ1, θ2, siter)
      avgβ1max += β1max
      d̂ .= - (∇fk .* (oneT - β1max) .+ momentum .* β1max) ./ (oneT - β1^siter)
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.Δ β1max
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

    step_underflow && set_status!(stats, :small_step)
    solver.Δ == 0 && set_status!(stats, :exception) # :small_step exception should happen before
    done = stats.status != :unknown
  end

  avgβ1max /= siter
  stats.solver_specific[:avgβ1max] = avgβ1max
  set_solution!(stats, x)
  return stats
end

"""
  solve_tadam_subproblem!(s, ∇fk, ̂d, ̂v, Δk, β1max, e)
Compute 
argmin -̂dᵀs + 0.5 sᵀ diag(sqrt.(̂v)+e) s
  s
s.t.   ||s||∞ <= Δk      
Stores the argmin in `s`.
"""
function solve_tadam_subproblem!(s::V, d̂::V, v̂::V, Δk::T, e::T) where {V, T}
  s .= min.(Δk , max.(-Δk , d̂ ./ (sqrt.(v̂) .+ e) ) )
end

"""
find_beta(m, mdot∇f, norm_∇f, β1, θ1, θ2)

Compute β1max which saturates the contibution of the momentum term to the gradient.
`β1max` is computed such that the two gradient-related conditions are ensured: 
1. ( (1-β1max) * ‖∇f(xk)‖² + β1max * ∇f(xk)ᵀm ) / (1-β1^(siter)) ≥ θ1 * ‖∇f(xk)‖²
2. ‖∇f(xk)‖ ≥ θ2 * ‖(1-β1max) * ∇f(xk) .+ β1max .* m‖ / (1-β1^(siter))
with `m` the momentum term and `mdot∇f = ∇f(xk)ᵀm` 
"""
function find_beta(p::V, mdot∇f::T, norm_∇f::T, β1::T, θ1::T, θ2::T, siter::Int) where {T, V}
  n1 = norm_∇f^2 - mdot∇f
  n2 = norm(p)
  b = (1-β1^(siter))
  β11 = n1 > 0 ? (1 - θ1*b) * norm_∇f^2 / n1 : β1
  β12 = n2 != 0 ? (1 - θ2*b) * norm_∇f / n2 : β1
  return min(β1, min(β11, β12))
end