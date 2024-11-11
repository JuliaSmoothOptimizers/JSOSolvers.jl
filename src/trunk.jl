export trunk, TrunkSolver, TRUNKParameterSet

trunk(nlp::AbstractNLPModel; variant = :Newton, kwargs...) = trunk(Val(variant), nlp; kwargs...)

# Default algorithm parameter values
const TRUNK_bk_max = DefaultParameter(10)
const TRUNK_monotone = DefaultParameter(true)
const TRUNK_nm_itmax = DefaultParameter(25)

"""
    TRUNKParameterSet <: AbstractParameterSet

This structure designed for `tron` regroups the following parameters:
  - `bk_max`: algorithm parameter.
  - `monotone`: algorithm parameter.
  - `nm_itmax`: algorithm parameter.

An additional constructor is

    TRUNKParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `bk_max::Int = $(TRUNK_bk_max)`
  - `monotone::Bool = $(TRUNK_monotone)`
  - `nm_itmax::Int = $(TRUNK_nm_itmax)`
"""
struct TRUNKParameterSet <: AbstractParameterSet
  bk_max::Parameter{Int, IntegerRange{Int}}
  monotone::Parameter{Bool, BinaryRange{Bool}}
  nm_itmax::Parameter{Int, IntegerRange{Int}}
end

# add a default constructor
function TRUNKParameterSet(
  nlp::AbstractNLPModel;
  bk_max::Int = get(TRUNK_bk_max, nlp),
  monotone::Bool = get(TRUNK_monotone, nlp),
  nm_itmax::Int = get(TRUNK_nm_itmax, nlp),
)
  TRUNKParameterSet(
    Parameter(bk_max, IntegerRange(1, typemax(Int))),
    Parameter(monotone, BinaryRange()),
    Parameter(nm_itmax, IntegerRange(1, typemax(Int))),
  )
end

"""
    trunk(nlp; kwargs...)

A trust-region solver for unconstrained optimization using exact second derivatives.

For advanced usage, first define a `TrunkSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TrunkSolver(nlp, subsolver_type::Type{<:KrylovSolver} = CgSolver)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `subsolver_logger::AbstractLogger = NullLogger()`: subproblem's logger.
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `bk_max::Int = $(TRUNK_bk_max)`: algorithm parameter, see [`TRUNKParameterSet`](@ref).
- `monotone::Bool = $(TRUNK_monotone)`: algorithm parameter, see [`TRUNKParameterSet`](@ref).
- `nm_itmax::Int = $(TRUNK_nm_itmax)`: algorithm parameter, see [`TRUNKParameterSet`](@ref).
- `verbose::Int = 0`: if > 0, display iteration information every `verbose` iteration.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver.
- `M`: linear operator that models a Hermitian positive-definite matrix of size `n`; passed to Krylov subsolvers. 

# Output
The returned value is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(Callback_docstring)

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
  Sub <: KrylovSolver{T, T, V},
  Op <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
  x::V
  xt::V
  gx::V
  gt::V
  gn::V
  Hs::V
  subsolver::Sub
  H::Op
  tr::TrustRegion{T, V}
  params::TRUNKParameterSet
end

function TrunkSolver(
  nlp::AbstractNLPModel{T, V};
  bk_max::Int = get(TRUNK_bk_max, nlp),
  monotone::Bool = get(TRUNK_monotone, nlp),
  nm_itmax::Int = get(TRUNK_nm_itmax, nlp),
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
) where {T, V <: AbstractVector{T}}
  params = TRUNKParameterSet(nlp; bk_max = bk_max, monotone = monotone, nm_itmax = nm_itmax)
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
  return TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, subsolver, H, tr, params)
end

function SolverCore.reset!(solver::TrunkSolver)
  solver.tr.good_grad = false
  solver.tr.radius = solver.tr.initial_radius
  solver
end

function SolverCore.reset!(solver::TrunkSolver, nlp::AbstractNLPModel)
  @assert (length(solver.gn) == 0) || isa(nlp, QuasiNewtonModel)
  solver.H = hess_op!(nlp, solver.x, solver.Hs)
  solver.tr.good_grad = false
  solver.tr.radius = solver.tr.initial_radius
  solver
end

@doc (@doc TrunkSolver) function trunk(
  ::Val{:Newton},
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  subsolver_type::Type{<:KrylovSolver} = CgSolver,
  kwargs...,
) where {V}
  solver = TrunkSolver(nlp; subsolver_type = subsolver_type)
  return solve!(solver, nlp; x = x, kwargs...)
end

function SolverCore.solve!(
  solver::TrunkSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  subsolver_logger::AbstractLogger = NullLogger(),
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Float64 = 30.0,
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  M = I,
) where {T, V <: AbstractVector{T}}
  if !(nlp.meta.minimize)
    error("trunk only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("trunk should only be called for unconstrained problems. Try tron instead")
  end

  # parameters
  bk_max = value(solver.params.bk_max)
  monotone = value(solver.params.monotone)
  nm_itmax = value(solver.params.nm_itmax)

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

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

  f = obj(nlp, x)
  grad!(nlp, x, ∇f)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇f)
  ∇fNorm2 = norm(∇f)
  ∇fNormM = normM!(n, ∇f, M, Hs)
  ϵ = atol + rtol * ∇fNorm2
  tr = solver.tr
  tr.radius = min(max(∇fNormM / 10, one(T)), T(100))

  # Non-monotone mode parameters.
  # fmin: current best overall objective value
  # nm_iter: number of successful iterations since fmin was first attained
  # fref: objective value at reference iteration
  # σref: cumulative model decrease over successful iterations since the reference iteration
  fmin = fref = fcur = f
  σref = σcur = zero(T)
  nm_iter = 0

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, ∇fNorm2)
  optimal = ∇fNorm2 ≤ ϵ
  fmin = min(-one(T), f) / eps(T)
  unbounded = f < fmin

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :inner, :bk, :cgstatus],
    [Int, T, T, T, T, Int, Int, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )
  verbose > 0 && @info log_row([stats.iter, f, ∇fNorm2, T, T, Int, Int, String])

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
    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖_M ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    cgtol = max(rtol, min(T(0.1), √∇fNormM, T(0.9) * cgtol))
    ∇f .*= -1
    Krylov.solve!(
      subsolver,
      H,
      ∇f,
      atol = atol,
      rtol = cgtol,
      radius = tr.radius,
      itmax = max(2 * n, 50),
      timemax = max_time - stats.elapsed_time,
      verbose = subsolver_verbose,
      M = M,
    )
    s, cg_stats = subsolver.x, subsolver.stats

    # Compute actual vs. predicted reduction.
    sNorm = nrm2(n, s)
    copyaxpy!(n, one(T), s, x, xt)
    slope = -dot(n, s, ∇f)
    mul!(Hs, H, s)
    curv = dot(n, s, Hs)
    Δq = slope + curv / 2
    ft = obj(nlp, xt)

    ared, pred = aredpred!(tr, nlp, f, ft, Δq, xt, s, slope)
    if pred ≥ 0
      stats.status = :neg_pred
      done = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
      if pred_hist ≥ 0
        stats.status = :neg_pred
        done = true
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
        stats.status = :not_desc
        done = true
        continue
      end
      α = one(T)
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= T(1.2)
        copyaxpy!(n, α, s, x, xt)
        ft = obj(nlp, xt)
      end
      sNorm *= α
      scal!(n, α, s)
      slope *= α
      Δq = slope + α * α * curv / 2
      ared, pred = aredpred!(tr, nlp, f, ft, Δq, xt, s, slope)
      if pred ≥ 0
        stats.status = :neg_pred
        done = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
        if pred_hist ≥ 0
          stats.status = :neg_pred
          done = true
          continue
        end
        ρ_hist = ared_hist / pred_hist
        tr.ratio = max(tr.ratio, ρ_hist)
      end
    end

    set_iter!(stats, stats.iter + 1)

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
      f = ft
      if !tr.good_grad
        grad!(nlp, x, ∇f)
      else
        ∇f .= tr.gt
        tr.good_grad = false
      end
      ∇fNorm2 = nrm2(n, ∇f)
      ∇fNormM = normM!(n, ∇f, M, Hs)

      set_objective!(stats, f)
      set_time!(stats, time() - start_time)
      set_dual_residual!(stats, ∇fNorm2)

      if isa(nlp, QuasiNewtonModel)
        ∇fn .-= ∇f
        ∇fn .*= -1  # = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, ∇fn)
        ∇fn .= ∇f
      end

      verbose > 0 &&
        mod(stats.iter, verbose) == 0 &&
        @info log_row([
          stats.iter,
          f,
          ∇fNorm2,
          tr.radius,
          tr.ratio,
          cg_stats.niter,
          bk,
          cg_stats.status,
        ])
    end

    # Move on.
    update!(tr, sNorm)

    optimal = ∇fNorm2 ≤ ϵ
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
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm2, tr.radius])

  set_solution!(stats, x)
  stats
end
