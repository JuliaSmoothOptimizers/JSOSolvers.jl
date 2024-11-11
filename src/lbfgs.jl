export lbfgs, LBFGSSolver, LBFGSParameterSet

# Default algorithm parameter values
const LBFGS_mem = DefaultParameter(5)
const LBFGS_τ₁ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.9999), "T(0.9999)")
const LBFGS_bk_max = DefaultParameter(25)

"""
    LBFGSParameterSet{T} <: AbstractParameterSet

This structure designed for `lbfgs` regroups the following parameters:
  - `mem`: memory parameter of the `lbfgs` algorithm
  - `τ₁`: slope factor in the Wolfe condition when performing the line search
  - `bk_max`: maximum number of backtracks when performing the line search.

An additional constructor is

    LBFGSParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `mem::Int = $(LBFGS_mem)`
  - `τ₁::T = $(LBFGS_τ₁)`
  - `bk_max:: Int = $(LBFGS_bk_max)`
"""
struct LBFGSParameterSet{T} <: AbstractParameterSet
  mem::Parameter{Int, IntegerRange{Int}}
  τ₁::Parameter{T, RealInterval{T}}
  bk_max::Parameter{Int, IntegerRange{Int}}
end

# add a default constructor
function LBFGSParameterSet(
  nlp::AbstractNLPModel{T};
  mem::Int = get(LBFGS_mem, nlp),
  τ₁::T = get(LBFGS_τ₁, nlp),
  bk_max::Int = get(LBFGS_bk_max, nlp),
) where {T}
  LBFGSParameterSet(
    Parameter(mem, IntegerRange(Int(5), Int(20))),
    Parameter(τ₁, RealInterval(T(0), T(1), lower_open = true)),
    Parameter(bk_max, IntegerRange(Int(1), Int(100))),
  )
end

"""
    lbfgs(nlp; kwargs...)

An implementation of a limited memory BFGS line-search method for unconstrained minimization.

For advanced usage, first define a `LBFGSSolver` to preallocate the memory used in the algorithm, and then call `solve!`.

    solver = LBFGSSolver(nlp; mem::Int = $(LBFGS_mem))
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `x::V = nlp.meta.x0`: the initial guess.
- `mem::Int = $(LBFGS_mem)`: algorithm parameter, see [`LBFGSParameterSet`](@ref).
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `τ₁::T = $(LBFGS_τ₁)`: algorithm parameter, see [`LBFGSParameterSet`](@ref).
- `bk_max:: Int = $(LBFGS_bk_max)`: algorithm parameter, see [`LBFGSParameterSet`](@ref).
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `verbose_subsolver::Int = 0`: if > 0, display iteration information every `verbose_subsolver` iteration of the subsolver.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(Callback_docstring)

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
solver = LBFGSSolver(nlp; mem = $(LBFGS_mem));
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
  params::LBFGSParameterSet{T}
end

function LBFGSSolver(nlp::M; kwargs...) where {T, V, M <: AbstractNLPModel{T, V}}
  nvar = nlp.meta.nvar

  params = LBFGSParameterSet(nlp; kwargs...)
  mem = value(params.mem)

  x = V(undef, nvar)
  d = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  H = InverseLBFGSOperator(T, nvar, mem = mem, scaling = true)
  h = LineModel(nlp, x, d)
  Op = typeof(H)
  return LBFGSSolver{T, V, Op, M}(x, xt, gx, gt, d, H, h, params)
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
  nlp::AbstractNLPModel{T, V};
  x::V = nlp.meta.x0,
  mem::Int = get(LBFGS_mem, nlp),
  τ₁::T = get(LBFGS_τ₁, nlp),
  bk_max::Int = get(LBFGS_bk_max, nlp),
  kwargs...,
) where {T, V}
  solver = LBFGSSolver(nlp; mem = mem, τ₁ = τ₁, bk_max = bk_max)
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

  # parameters
  τ₁ = value(solver.params.τ₁)
  bk_max = value(solver.params.bk_max)

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
  fmin = min(-one(T), f) / eps(T)
  unbounded = f < fmin

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
    unbounded = f < fmin

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
  end
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm])

  set_solution!(stats, x)
  stats
end
