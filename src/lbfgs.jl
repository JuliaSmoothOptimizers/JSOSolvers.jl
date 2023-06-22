export lbfgs, LBFGSSolver

"""
    lbfgs(nlp; kwargs...)

An implementation of a limited memory BFGS line-search method for unconstrained minimization.

For advanced usage, first define a `LBFGSSolver` to preallocate the memory used in the algorithm, and then call `solve!`.

    solver = LBFGSSolver(nlp; mem::Int = 5)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `x::V = nlp.meta.x0`: the initial guess.
- `mem::Int = 5`: memory parameter of the `lbfgs` algorithm.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `τ₁::T = T(0.9999)`: slope factor in the Wolfe condition when performing the line search.
- `bk_max:: Int = 25`: maximum number of backtracks when performing the line search.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `verbose_subsolver::Int = 0`: if > 0, display iteration information every `verbose_subsolver` iteration of the subsolver.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

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
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has found a stopping criteria. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
stats = lbfgs(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
solver = LBFGSSolver(nlp; mem = 5);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct LBFGSSolver{T, V, Op <: AbstractLinearOperator{T}, M <: AbstractNLPModel{T, V}} <:
               AbstractOptimizationSolver
  x::V
  xt::V
  gx::V
  gt::V
  d::V
  H::Op
  h::LineModel{T, V, M}
end

function LBFGSSolver(nlp::M; mem::Int = 5) where {T, V, M <: AbstractNLPModel{T, V}}
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  d = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  H = InverseLBFGSOperator(T, nvar, mem = mem, scaling = true)
  h = LineModel(nlp, x, d)
  Op = typeof(H)
  return LBFGSSolver{T, V, Op, M}(x, xt, gx, gt, d, H, h)
end

function SolverCore.reset!(solver::LBFGSSolver)
  reset!(solver.H)
end

function SolverCore.reset!(solver::LBFGSSolver, nlp::AbstractNLPModel)
  reset!(solver.H)
  solver.h = LineModel(nlp, solver.x, solver.d)
  solver
end

@doc (@doc LBFGSSolver) function lbfgs(
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  mem::Int = 5,
  kwargs...,
) where {V}
  solver = LBFGSSolver(nlp; mem = mem)
  return solve!(solver, nlp; x = x, kwargs...)
end

function SolverCore.solve!(
  solver::LBFGSSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Float64 = 30.0,
  τ₁::T = T(0.9999),
  bk_max::Int = 25,
  verbose::Int = 0,
  verbose_subsolver::Int = 0,
) where {T, V}
  if !(nlp.meta.minimize)
    error("lbfgs only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("lbfgs should only be called for unconstrained problems. Try tron instead")
  end

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.meta.nvar

  solver.x .= x
  x = solver.x
  xt = solver.xt
  ∇f = solver.gx
  ∇ft = solver.gt
  d = solver.d
  h = solver.h
  H = solver.H
  reset!(H)

  f, ∇f = objgrad!(nlp, x, ∇f)

  ∇fNorm = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, ∇fNorm)

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :slope, :bk],
    [Int, T, T, T, Int],
    hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :slope => "∇fᵀd"),
  )
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm, T, Int])

  optimal = ∇fNorm ≤ ϵ

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

  while !done
    mul!(d, H, ∇f, -one(T), zero(T))
    slope = dot(n, d, ∇f)
    if slope ≥ 0
      @error "not a descent direction" slope
      set_status!(stats, :not_desc)
      done = true
      continue
    end

    # Perform improved Armijo linesearch.
    t, good_grad, ft, nbk, nbW =
      armijo_wolfe(h, f, slope, ∇ft, τ₁ = τ₁, bk_max = bk_max, verbose = Bool(verbose_subsolver))

    copyaxpy!(n, t, d, x, xt)
    good_grad || grad!(nlp, xt, ∇ft)

    # Update L-BFGS approximation.
    d .*= t
    @. ∇f = ∇ft - ∇f
    push!(H, d, ∇f)

    # Move on.
    x .= xt
    f = ft
    ∇f .= ∇ft

    ∇fNorm = nrm2(n, ∇f)

    set_iter!(stats, stats.iter + 1)

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row(Any[stats.iter, f, ∇fNorm, slope, nbk])

    set_objective!(stats, f)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, ∇fNorm)
    optimal = ∇fNorm ≤ ϵ

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
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm])

  set_solution!(stats, x)
  stats
end
