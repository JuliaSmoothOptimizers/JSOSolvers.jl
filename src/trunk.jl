export trunk, TrunkSolver

trunk(nlp::AbstractNLPModel; variant = :Newton, kwargs...) = trunk(Val(variant), nlp; kwargs...)

"""
    trunk(nlp; kwargs...)

A trust-region solver for unconstrained optimization using exact second derivatives.

For advanced usage, first define a `TrunkSolver` to preallocate the memory used in the algorithm, and then call `solve!`.

    solver = TrunkSolver(nlp, subsolver_type::Type{<:KrylovSolver} = CgSolver)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `subsolver_logger::AbstractLogger = NullLogger()`: subproblem's logger.
- `callback = (nlp, solver) -> nothing`: callback function called after each iteration. See the Callback section below.
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `bk_max::Int = 10`: algorithm parameter.
- `monotone::Bool = true`: algorithm parameter.
- `nm_itmax::Int = 25`: algorithm parameter.
- `verbose::Int = 0`: if > 0, display iteration information every `verbose` iteration.
- `verbose_subsolver::Int = 0`: if > 0, display iteration information every `verbose_subsolver` iteration of the subsolver.

## Callback

The callback is called after each iteration.
The expected signature of callback is `(nlp, solver)`, and its output is ignored.
Notice that changing any of the input arguments will affect the subsequent iterations.
In particular, setting `solver.output.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:

- `solver.x`: current iteration.
- `solver.gx`: current gradient.
- `solver.output`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `solver.output.dual_feas`: norm of current gradient.
  - `solver.output.iter`: current iteration counter.
  - `solver.output.objective`: current objective function value.
  - `solver.output.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has found a stopping criteria. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `solver.output.elapsed_time`: elapsed time in seconds.

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# References
This implementation follows the description given in

    A. R. Conn, N. I. M. Gould, and Ph. L. Toint,
    Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.
    SIAM, Philadelphia, USA, 2000.
    DOI: 10.1137/1.9780898719857

The main algorithm follows the basic trust-region method described in Section 6.
The backtracking linesearch follows Section 10.3.2.
The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.

# Examples
```jldoctest; output = false
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = trunk(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest; output = false
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = TrunkSolver(nlp)
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct TrunkSolver{
  T,
  V <: AbstractVector{T},
  Sub <: KrylovSolver{T, V},
  Op <: AbstractLinearOperator{T},
} <: AbstractOptSolver{T, V}
  x::V
  xt::V
  gx::V
  gt::V
  gn::V
  Hs::V
  subsolver::Sub
  H::Op
  tr::TrustRegion{T, V}
  output::GenericExecutionStats{T, V}
end

function TrunkSolver(
  nlp::AbstractNLPModel{T, V};
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
) where {T, V <: AbstractVector{T}}
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  subsolver = subsolver_type(nvar, nvar, V)
  Sub = typeof(subsolver)
  H = hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  output = GenericExecutionStats(:unknown, nlp, solution = x)
  return TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr, output)
end

function LinearOperators.reset!(::TrunkSolver) end

@doc (@doc TrunkSolver) function trunk(
  ::Val{:Newton},
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
  kwargs...,
) where {V}
  solver = TrunkSolver(nlp, subsolver_type = subsolver_type)
  return solve!(solver, nlp; x = x, kwargs...)
end

function solve!(
  solver::TrunkSolver{T, V},
  nlp::AbstractNLPModel{T, V};
  subsolver_logger::AbstractLogger = NullLogger(),
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_eval::Int = -1,
  max_time::Float64 = 30.0,
  bk_max::Int = 10,
  monotone::Bool = true,
  nm_itmax::Int = 25,
  verbose::Int = 0,
  verbose_subsolver::Int = 0,
) where {T, V <: AbstractVector{T}}
  if !(nlp.meta.minimize)
    error("trunk only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("trunk should only be called for unconstrained problems. Try tron instead")
  end

  output = solver.output

  start_time = time()
  output.elapsed_time = 0.0

  n = nlp.meta.nvar

  solver.x .= x
  x = solver.x
  xt = solver.xt
  ∇f = solver.gx
  ∇fn = solver.gn
  Hs = solver.Hs
  subsolver = solver.subsolver
  H = solver.H
  tr = solver.tr

  cgtol = one(T)  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = eps(T)^T(1 / 4)

  output.iter = 0
  output.objective = obj(nlp, x)
  grad!(nlp, x, ∇f)
  ∇fNorm2 = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm2
  tr = solver.tr
  tr.radius = min(max(∇fNorm2 / 10, one(T)), T(100))

  # Non-monotone mode parameters.
  # fmin: current best overall objective value
  # nm_iter: number of successful iterations since fmin was first attained
  # fref: objective value at reference iteration
  # σref: cumulative model decrease over successful iterations since the reference iteration
  fmin = fref = fcur = output.objective
  σref = σcur = zero(T)
  nm_iter = 0

  optimal = ∇fNorm2 ≤ ϵ
  output.dual_feas = ∇fNorm2

  stalled = false
  output.status = get_status(
    nlp,
    elapsed_time = output.elapsed_time,
    optimal = optimal,
    max_eval = max_eval,
    max_time = max_time,
  )

  callback(nlp, solver)

  done = output.status != :unknown

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :inner, :bk, :cgstatus],
    [Int, T, T, T, T, Int, Int, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )

  while !done
    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    cgtol = max(rtol, min(T(0.1), √∇fNorm2, T(0.9) * cgtol))
    solve!(
      subsolver,
      H,
      ∇f,
      atol = atol,
      rtol = cgtol,
      radius = tr.radius,
      itmax = max(2 * n, 50),
      verbose = verbose_subsolver,
    )
    s, cg_stats = subsolver.x, subsolver.stats

    # Compute actual vs. predicted reduction.
    sNorm = nrm2(n, s)
    @. s = -s
    copyaxpy!(n, one(T), s, x, xt)
    slope = dot(n, s, ∇f)
    mul!(Hs, H, s)
    curv = dot(n, s, Hs)
    Δq = slope + curv / 2
    ft = obj(nlp, xt)

    ared, pred = aredpred!(tr, nlp, output.objective, ft, Δq, xt, s, slope)
    if pred ≥ 0
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
      if pred_hist ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      ρ_hist = ared_hist / pred_hist
      tr.ratio = max(tr.ratio, ρ_hist)
    end

    bk = 0
    if !acceptable(tr)
      # Perform backtracking linesearch along s
      # Scaling s to the trust-region boundary, as recommended in
      # Algorithm 10.3.2 of the Trust-Region book
      # appears to deteriorate results.
      # BLAS.scal!(n, tr.radius / sNorm, s, 1)
      # slope *= tr.radius / sNorm
      # sNorm = tr.radius

      if slope ≥ 0
        @error "not a descent direction: slope = $slope, ‖∇f‖ = $∇fNorm2"
        status = :not_desc
        stalled = true
        continue
      end
      α = one(T)
      while (bk < bk_max) && (ft > output.objective + β * α * slope)
        bk = bk + 1
        α /= T(1.2)
        copyaxpy!(n, α, s, x, xt)
        ft = obj(nlp, xt)
      end
      sNorm *= α
      scal!(n, α, s)
      slope *= α
      Δq = slope + α * α * curv / 2
      ared, pred = aredpred!(tr, nlp, output.objective, ft, Δq, xt, s, slope)
      if pred ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
        if pred_hist ≥ 0
          status = :neg_pred
          stalled = true
          continue
        end
        ρ_hist = ared_hist / pred_hist
        tr.ratio = max(tr.ratio, ρ_hist)
      end
    end

    verbose > 0 &&
      mod(iter, verbose) == 0 &&
      @info log_row([
        output.iter,
        output.objective,
        ∇fNorm2,
        tr.radius,
        tr.ratio,
        length(cg_stats.residuals),
        bk,
        cg_stats.status,
      ])
    output.iter += 1

    if acceptable(tr)
      # Update non-monotone mode parameters.
      if !monotone
        σref = σref + Δq
        σcur = σcur + Δq
        if ft < fmin
          # New overall best objective value found.
          fcur = ft
          fmin = ft
          σcur = zero(T)
          nm_iter = 0
        else
          nm_iter = nm_iter + 1

          if ft > fcur
            fcur = ft
            σcur = zero(T)
          end

          if nm_iter ≥ nm_itmax
            fref = fcur
            σref = σcur
          end
        end
      end

      x .= xt
      output.objective = ft
      if !tr.good_grad
        grad!(nlp, x, ∇f)
      else
        ∇f .= tr.gt
        tr.good_grad = false
      end
      ∇fNorm2 = nrm2(n, ∇f)

      if isa(nlp, QuasiNewtonModel)
        ∇fn .-= ∇f
        @. ∇fn = -∇fn  # = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, ∇fn)
        ∇fn .= ∇f
      end
    end

    # Move on.
    update!(tr, sNorm)

    optimal = ∇fNorm2 ≤ ϵ
    output.elapsed_time = time() - start_time
    output.dual_feas = ∇fNorm2

    output.status = get_status(
      nlp,
      elapsed_time = output.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      max_time = max_time,
    )

    callback(nlp, solver)

    done = output.status != :unknown
  end
  verbose > 0 && @info log_row(Any[output.iter, output.objective, ∇fNorm2, tr.radius])
  output.dual_feas = ∇fNorm2


  return output
end
