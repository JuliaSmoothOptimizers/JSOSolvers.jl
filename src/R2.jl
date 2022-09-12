export R2, R2Solver

"""
    R2(nlp; kwargs...)
    solver = R2Solver(nlp;)
    solve!(solver, nlp; kwargs...)
    A first-order quadratic regularization method for unconstrained optimization
# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.
# Keyword arguments 
- x0::V = nlp.meta.x0`: the initial guess
- atol = eps(T)^(1 / 3): absolute tolerance
- rtol = eps(T)^(1 / 3): relative tolerance: algorithm stop when ||∇f(x)|| ≤ ϵ_abs + ϵ_rel*||∇f(x0)||
- η1 = eps(T)^(1/4), η2 = T(0.95): step acceptance parameters
- γ1 = T(1/2), γ2 = 1/γ1: regularization update parameters
- σmin = eps(T): step parameter for R2 algorithm
- max_eval::Int: maximum number of evaluation of the objective function
- max_time::Float64 = 3600.0: maximum time limit in seconds
- verbose::Int = 0: if > 0, display iteration details every `verbose` iteration.
# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.
# Callback
The callback is called after each iteration.
The expected signature of the callback is `callback(nlp, solver)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `solver.output.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `solver.output`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `solver.output.dual_feas`: norm of current gradient;
  - `solver.output.iter`: current iteration counter;
  - `solver.output.objective`: current objective function value;
  - `solver.output.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has found a stopping criteria. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `solver.output.elapsed_time`: elapsed time in seconds.
# Examples
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
solver = R2Solver(nlp);
stats = solve!(solver, nlp)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct R2Solver{T, V}
  x::V
  gx::V
  cx::V
  stats::GenericExecutionStats{T, V}
end

function R2Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  stats = GenericExecutionStats(nlp)
  return R2Solver{T, V}(x, gx, cx, stats)
end

@doc (@doc R2Solver) function R2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = R2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function solve!(
  solver::R2Solver{T, V},
  nlp::AbstractNLPModel{T, V};
  callback = (args...) -> nothing,
  x0::V = nlp.meta.x0,
  atol = eps(T)^(1 / 2),
  rtol = eps(T)^(1 / 2),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  γ1 = T(1 / 2),
  γ2 = 1 / γ1,
  σmin = zero(T),
  max_time::Float64 = 3600.0,
  max_eval::Int = -1,
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("R2 should only be called on unconstrained problems.")

  stats = solver.stats
  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x0
  ∇fk = solver.gx
  ck = solver.cx

  set_iter!(stats, 0)
  set_solution!(stats, x)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_residuals!(stats, zero(T), norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1))

  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
  end
  if verbose > 0 && mod(iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
  end

  set_status!(
    stats,
    get_status(
    nlp,
    elapsed_time = stats.elapsed_time,
    optimal = optimal,
    max_eval = max_eval,
    max_time = max_time,
  )
  )

  callback(nlp, solver)

  done =
    (stats.status == :first_order) ||
    (stats.status == :max_eval) ||
    (stats.status == :max_time) ||
    (stats.status == :user)

  while !done
    ck .= x .- (∇fk ./ σk)
    ΔTk = norm_∇fk^2 / σk
    fck = obj(nlp, ck)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    ρk = (stats.objective - fck) / ΔTk

    # Update regularization parameters
    if ρk >= η2
      σk = max(σmin, γ1 * σk)
    elseif ρk < η1
      σk = σk * γ2
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= ck
      set_solution!(stats, x)
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_residuals!(stats, zero(T), norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇fk σk
    end

    set_status!(
      stats,
      get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      max_time = max_time,
    )
    )

    callback(nlp, solver)

    done =
      (stats.status == :first_order) ||
      (stats.status == :max_eval) ||
      (stats.status == :max_time) ||
      (stats.status == :user)
  end

  return stats
end
