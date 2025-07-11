export TrunkSolverNLS, TRUNKLSParameterSet

const trunkls_allowed_subsolvers = (:cgls, :crls, :lsqr, :lsmr)

trunk(nlp::AbstractNLSModel; variant = :GaussNewton, kwargs...) =
  trunk(Val(variant), nlp; kwargs...)

# Default algorithm parameter values
const TRUNKLS_bk_max = DefaultParameter(10)
const TRUNKLS_monotone = DefaultParameter(true)
const TRUNKLS_nm_itmax = DefaultParameter(25)

"""
    TRUNKLSParameterSet <: AbstractParameterSet

This structure designed for `tron` regroups the following parameters:
  - `bk_max`: algorithm parameter.
  - `monotone`: algorithm parameter.
  - `nm_itmax`: algorithm parameter.

An additional constructor is

    TRUNKLSParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `bk_max::Int = $(TRUNKLS_bk_max)`
  - `monotone::Bool = $(TRUNKLS_monotone)`
  - `nm_itmax::Int = $(TRUNKLS_nm_itmax)`
"""
struct TRUNKLSParameterSet <: AbstractParameterSet
  bk_max::Parameter{Int, IntegerRange{Int}}
  monotone::Parameter{Bool, BinaryRange{Bool}}
  nm_itmax::Parameter{Int, IntegerRange{Int}}
end

# add a default constructor
function TRUNKLSParameterSet(
  nlp::AbstractNLPModel;
  bk_max::Int = get(TRUNKLS_bk_max, nlp),
  monotone::Bool = get(TRUNKLS_monotone, nlp),
  nm_itmax::Int = get(TRUNKLS_nm_itmax, nlp),
)
  TRUNKLSParameterSet(
    Parameter(bk_max, IntegerRange(1, typemax(Int))),
    Parameter(monotone, BinaryRange()),
    Parameter(nm_itmax, IntegerRange(1, typemax(Int))),
  )
end

"""
    trunk(nls; kwargs...)

A pure Julia implementation of a trust-region solver for nonlinear least-squares problems:

    min ½‖F(x)‖²

For advanced usage, first define a `TrunkSolverNLS` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TrunkSolverNLS(nls, subsolver::Symbol = :lsmr)
    solve!(solver, nls; kwargs...)

# Arguments
- `nls::AbstractNLSModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `Fatol::T = √eps(T)`: absolute tolerance on the residual.
- `Frtol::T = eps(T)`: relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `bk_max::Int = $(TRUNKLS_bk_max)`: algorithm parameter, see [`TRUNKLSParameterSet`](@ref).
- `monotone::Bool = $(TRUNKLS_monotone)`: algorithm parameter, see [`TRUNKLSParameterSet`](@ref).
- `nm_itmax::Int = $(TRUNKLS_nm_itmax)`: algorithm parameter, see [`TRUNKLSParameterSet`](@ref).
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver.

See `JSOSolvers.trunkls_allowed_subsolvers` for a list of available Krylov solvers.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

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
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2)
stats = trunk(nls)
# output
"Execution stats: first-order stationary"
```

```jldoctest; output = false
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2)
solver = TrunkSolverNLS(nls)
stats = solve!(solver, nls)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct TrunkSolverNLS{
  T,
  V <: AbstractVector{T},
  Sub <: KrylovWorkspace{T, T, V},
  Op <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
  x::V
  xt::V
  temp::V
  gx::V
  gt::V
  tr::TrustRegion{T, V}
  Fx::V
  rt::V
  Av::V
  Atv::V
  A::Op
  krylov_subsolver::Sub
  params::TRUNKLSParameterSet
end

function TrunkSolverNLS(
  nlp::AbstractNLPModel{T, V};
  bk_max::Int = get(TRUNKLS_bk_max, nlp),
  monotone::Bool = get(TRUNKLS_monotone, nlp),
  nm_itmax::Int = get(TRUNKLS_nm_itmax, nlp),
  subsolver::Symbol = :lsmr,
) where {T, V <: AbstractVector{T}}
  params = TRUNKLSParameterSet(nlp; bk_max = bk_max, monotone = monotone, nm_itmax = nm_itmax)
  subsolver in trunkls_allowed_subsolvers ||
    error("subproblem solver must be one of $(trunkls_allowed_subsolvers)")

  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ
  x = V(undef, nvar)
  xt = V(undef, nvar)
  temp = V(undef, nequ)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  tr = TrustRegion(gt, one(T))

  Fx = V(undef, nequ)
  rt = V(undef, nequ)
  Av = V(undef, nequ)
  Atv = V(undef, nvar)
  A = jac_op_residual!(nlp, x, Av, Atv)
  Op = typeof(A)

  krylov_subsolver = krylov_workspace(Val(subsolver), nequ, nvar, V)
  Sub = typeof(krylov_subsolver)

  return TrunkSolverNLS{T, V, Sub, Op}(
    x,
    xt,
    temp,
    gx,
    gt,
    tr,
    rt,
    Fx,
    Av,
    Atv,
    A,
    krylov_subsolver,
    params,
  )
end

function SolverCore.reset!(solver::TrunkSolverNLS)
  solver.tr.good_grad = false
  solver.tr.radius = solver.tr.initial_radius
  solver
end
function SolverCore.reset!(solver::TrunkSolverNLS, nlp::AbstractNLSModel)
  solver.A = jac_op_residual!(nlp, solver.x, solver.Av, solver.Atv)
  solver.tr.good_grad = false
  solver.tr.radius = solver.tr.initial_radius
  solver
end

@doc (@doc TrunkSolverNLS) function trunk(
  ::Val{:GaussNewton},
  nlp::AbstractNLSModel;
  x::V = nlp.meta.x0,
  subsolver::Symbol = :lsmr,
  kwargs...,
) where {V}
  solver = TrunkSolverNLS(nlp; subsolver)
  return solve!(solver, nlp; x = x, kwargs...)
end

function SolverCore.solve!(
  solver::TrunkSolverNLS{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::Real = √eps(T),
  rtol::Real = √eps(T),
  Fatol::T = zero(T),
  Frtol::T = zero(T),
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Float64 = 30.0,
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
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

  SolverCore.reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  solver.x .= x
  x = solver.x
  ∇f = solver.gx
  krylov_subsolver = solver.krylov_subsolver

  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ
  cgtol = one(T)  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = eps(T)^T(1 / 4)

  r, rt = solver.Fx, solver.rt
  residual!(nlp, x, r)
  f, ∇f = objgrad!(nlp, x, ∇f, r, recompute = false)

  # preallocate storage for products with A and A'
  A = solver.A # jac_op_residual!(nlp, x, Av, Atv)
  mul!(∇f, A', r)
  ∇fNorm2 = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm2
  ϵF = Fatol + Frtol * 2 * √f
  tr = solver.tr
  tr.radius = min(max(∇fNorm2 / 10, one(T)), T(100))

  # Non-monotone mode parameters.
  # fmin: current best overall objective value
  # nm_iter: number of successful iterations since fmin was first attained
  # fref: objective value at reference iteration
  # σref: cumulative model decrease over successful iterations since the reference iteration
  fmin = fref = fcur = f
  σref = σcur = zero(T)
  nm_iter = 0

  # Preallocate xt.
  xt = solver.xt
  temp = solver.temp

  optimal = ∇fNorm2 ≤ ϵ
  small_residual = 2 * √f ≤ ϵF

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, ∇fNorm2)

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :radius, :step, :ratio, :inner, :bk, :cgstatus],
    [Int, T, T, T, T, T, Int, Int, String],
    hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :radius => "Δ"),
  )
  verbose > 0 && @info log_row([stats.iter, f, ∇fNorm2, T, T, T, Int, Int, String])

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      small_residual = small_residual,
      max_eval = max_eval,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
    temp .= .-r
    krylov_solve!(
      krylov_subsolver,
      A,
      temp,
      atol = atol,
      rtol = cgtol,
      radius = tr.radius,
      itmax = max(2 * (n + m), 50),
      timemax = max_time - stats.elapsed_time,
      verbose = subsolver_verbose,
    )
    s, cg_stats = krylov_subsolver.x, krylov_subsolver.stats

    # Compute actual vs. predicted reduction.
    sNorm = nrm2(n, s)
    copyaxpy!(n, one(T), s, x, xt)
    # slope = dot(∇f, s)
    mul!(temp, A, s)
    slope = dot(r, temp)
    curv = dot(temp, temp)
    Δq = slope + curv / 2
    residual!(nlp, xt, rt)
    ft = obj(nlp, x, rt, recompute = false)

    ared, pred = aredpred!(tr, nlp, rt, f, ft, Δq, xt, s, slope)
    if pred ≥ 0
      stats.status = :neg_pred
      done = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred!(tr, nlp, rt, fref, ft, σref + Δq, xt, s, slope)
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
        @error "not a descent direction" slope ∇fNorm2 sNorm
        stats.status = :not_desc
        done = true
        continue
      end
      α = one(T)
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= T(1.2)
        copyaxpy!(n, α, s, x, xt)
        residual!(nlp, xt, rt)
        ft = obj(nlp, x, rt, recompute = false)
      end
      sNorm *= α
      scal!(n, α, s)
      slope *= α
      Δq = slope + α * α * curv / 2
      ared, pred = aredpred!(tr, nlp, rt, f, ft, Δq, xt, s, slope)
      if pred ≥ 0
        stats.status = :neg_pred
        done = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = aredpred!(tr, nlp, rt, fref, ft, σref + Δq, xt, s, slope)
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

          if nm_iter >= nm_itmax
            fref = fcur
            σref = σcur
          end
        end
      end

      x .= xt # update A implicitly
      r .= rt
      f = ft
      if tr.good_grad
        ∇f .= tr.gt
        tr.good_grad = false
      else
        grad!(nlp, x, ∇f, r, recompute = false)
      end
      ∇fNorm2 = nrm2(n, ∇f)
    end

    # Move on.
    update!(tr, sNorm)

    set_objective!(stats, f)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, ∇fNorm2)

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row([
        stats.iter,
        f,
        ∇fNorm2,
        tr.radius,
        sNorm,
        tr.ratio,
        length(cg_stats.residuals),
        bk,
        cg_stats.status,
      ])

    optimal = ∇fNorm2 ≤ ϵ
    small_residual = 2 * √f ≤ ϵF

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        small_residual = small_residual,
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
