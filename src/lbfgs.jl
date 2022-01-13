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
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `mem::Int = 5`: memory parameter of the `lbfgs` algorithm.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

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
               AbstractOptSolver{T, V}
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

function LinearOperators.reset!(solver::LBFGSSolver)
  reset!(solver.H)
end

@doc (@doc LBFGSSolver) function lbfgs(
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  kwargs...,
) where {V}
  solver = LBFGSSolver(nlp)
  return solve!(solver, nlp; x = x, kwargs...)
end

function solve!(
  solver::LBFGSSolver{T, V},
  nlp::AbstractNLPModel{T, V};
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_eval::Int = -1,
  max_time::Float64 = 30.0,
  verbose::Int = 0,
  mem::Int = 5,
) where {T, V}
  if !(nlp.meta.minimize)
    error("lbfgs only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("lbfgs should only be called for unconstrained problems. Try tron instead")
  end

  start_time = time()
  elapsed_time = 0.0

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

  f = obj(nlp, x)
  grad!(nlp, x, ∇f)

  ∇fNorm = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm
  iter = 0

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :slope, :bk],
    [Int, T, T, T, Int],
    hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :slope => "∇fᵀd"),
  )

  optimal = ∇fNorm ≤ ϵ
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  stalled = false
  status = :unknown

  while !(optimal || tired || stalled)
    mul!(d, H, ∇f, -one(T), zero(T))
    slope = dot(n, d, ∇f)
    if slope ≥ 0
      @error "not a descent direction" slope
      status = :not_desc
      stalled = true
      continue
    end

    # Perform improved Armijo linesearch.
    t, good_grad, ft, nbk, nbW =
      armijo_wolfe(h, f, slope, ∇ft, τ₁ = T(0.9999), bk_max = 25, verbose = false)

    verbose > 0 && mod(iter, verbose) == 0 && @info log_row(Any[iter, f, ∇fNorm, slope, nbk])

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
    iter = iter + 1

    optimal = ∇fNorm ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  end
  verbose > 0 && @info log_row(Any[iter, f, ∇fNorm])

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_eval ≥ 0
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = f,
    dual_feas = ∇fNorm,
    iter = iter,
    elapsed_time = elapsed_time,
  )
end
