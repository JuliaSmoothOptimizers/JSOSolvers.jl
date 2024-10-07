#  Some parts of this code were adapted from
# https://github.com/PythonOptimizers/NLP.py/blob/develop/nlp/optimize/tron.py

export tron, TronSolver, TRONParameterSet

tron(nlp::AbstractNLPModel; variant = :Newton, kwargs...) = tron(Val(variant), nlp; kwargs...)

# Default algorithm parameter values
const TRON_μ₀ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1 // 100), "T(1 / 100)")
const TRON_μ₁ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1), "T(1)")
const TRON_σ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(10), "T(10)")

"""
    TRONParameterSet{T} <: AbstractParameterSet

This structure designed for `tron` regroups the following parameters:
  - `μ₀::T`: algorithm parameter in (0, 0.5).
  - `μ₁::T`: algorithm parameter in (0, +∞).
  - `σ::T`: algorithm parameter in (1, +∞).

An additional constructor is

    TRONParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `μ₀::T = $(TRON_μ₀)`
  - `μ₁::T = $(TRON_μ₁)`
  - `σ::T = $(TRON_σ)`
"""
struct TRONParameterSet{T} <: AbstractParameterSet
  μ₀::Parameter{T, RealInterval{T}}
  μ₁::Parameter{T, RealInterval{T}}
  σ::Parameter{T, RealInterval{T}}
end

# add a default constructor
function TRONParameterSet(
  nlp::AbstractNLPModel{T};
  μ₀::T = get(TRON_μ₀, nlp),
  μ₁::T = get(TRON_μ₁, nlp),
  σ::T = get(TRON_σ, nlp),
) where {T}
  TRONParameterSet(
    Parameter(μ₀, RealInterval(T(0), T(1 // 2), lower_open = true)),
    Parameter(μ₁, RealInterval(T(0), T(Inf), lower_open = true)),
    Parameter(σ, RealInterval(T(1), T(Inf), lower_open = true)),
  )
end

"""
    tron(nlp; kwargs...)

A pure Julia implementation of a trust-region solver for bound-constrained optimization:
    
        min f(x)    s.t.    ℓ ≦ x ≦ u
    
For advanced usage, first define a `TronSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TronSolver(nlp; kwargs...)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `x::V = nlp.meta.x0`: the initial guess.
- `μ₀::T = $(TRON_μ₀)`: algorithm parameter, see [`TRONParameterSet`](@ref).
- `μ₁::T = $(TRON_μ₁)`: algorithm parameter, see [`TRONParameterSet`](@ref).
- `σ::T = $(TRON_σ)`: algorithm parameter, see [`TRONParameterSet`](@ref).
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `max_cgiter::Int = 50`: subproblem's iteration limit.
- `use_only_objgrad::Bool = false`: If `true`, the algorithm uses only the function `objgrad` instead of `obj` and `grad`.
- `cgtol::T = T(0.1)`: subproblem tolerance.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖x - Proj(x - ∇f(xᵏ))‖ ≤ atol + rtol * ‖∇f(x⁰)‖. Proj denotes here the projection over the bounds.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver.

The keyword arguments of `TronSolver` are passed to the [`TRONTrustRegion`](https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/main/src/trust-region/tron-trust-region.jl) constructor.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(Callback_docstring)

# References
TRON is described in

    Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
    Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
    DOI: 10.1137/S1052623498345075

# Examples
```jldoctest; output = false
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));
stats = tron(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest; output = false
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));
solver = TronSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct TronSolver{
  T,
  V <: AbstractVector{T},
  Op <: AbstractLinearOperator{T},
  Aop <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
  x::V
  xc::V
  temp::V
  gx::V
  gt::V
  gn::V
  gpx::V
  s::V
  Hs::V
  H::Op
  tr::TRONTrustRegion{T, V}

  ifix::BitVector

  cg_solver::CgSolver{T, T, V}
  cg_rhs::V
  cg_op_diag::V
  cg_op::LinearOperator{T}

  ZHZ::Aop
  params::TRONParameterSet{T}
end

function TronSolver(
  nlp::AbstractNLPModel{T, V};
  μ₀::T = get(TRON_μ₀, nlp),
  μ₁::T = get(TRON_μ₁, nlp),
  σ::T = get(TRON_σ, nlp),
  max_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)),
  kwargs...,
) where {T, V <: AbstractVector{T}}
  params = TRONParameterSet(nlp; μ₀ = μ₀, μ₁ = μ₁, σ = σ)
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xc = V(undef, nvar)
  temp = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  gpx = V(undef, nvar)
  s = V(undef, nvar)
  Hs = V(undef, nvar)
  H = hess_op!(nlp, xc, Hs)
  Op = typeof(H)
  tr = TRONTrustRegion(gt, min(one(T), max_radius - eps(T)); max_radius = max_radius, kwargs...)

  ifix = BitVector(undef, nvar)

  cg_rhs = V(undef, nvar)
  cg_op_diag = V(undef, nvar)
  cg_op = opDiagonal(cg_op_diag)

  ZHZ = cg_op' * H * cg_op
  cg_solver = CgSolver(ZHZ, Hs)
  return TronSolver{T, V, Op, typeof(ZHZ)}(
    x,
    xc,
    temp,
    gx,
    gt,
    gn,
    gpx,
    s,
    Hs,
    H,
    tr,
    ifix,
    cg_solver,
    cg_rhs,
    cg_op_diag,
    cg_op,
    ZHZ,
    params,
  )
end

function SolverCore.reset!(solver::TronSolver)
  solver.tr.good_grad = false
  solver
end

function SolverCore.reset!(solver::TronSolver, nlp::AbstractNLPModel)
  @assert (length(solver.gn) == 0) || isa(nlp, QuasiNewtonModel)
  solver.H = hess_op!(nlp, solver.xc, solver.Hs)
  solver.ZHZ = solver.cg_op' * solver.H * solver.cg_op
  solver.tr.good_grad = false
  solver
end

@doc (@doc TronSolver) function tron(
  ::Val{:Newton},
  nlp::AbstractNLPModel{T, V};
  x::V = nlp.meta.x0,
  μ₀::T = get(TRON_μ₀, nlp),
  μ₁::T = get(TRON_μ₁, nlp),
  σ::T = get(TRON_σ, nlp),
  kwargs...,
) where {T, V}
  dict = Dict(kwargs)
  subsolver_keys = intersect(keys(dict), tron_keys)
  subsolver_kwargs = Dict(k => dict[k] for k in subsolver_keys)
  solver = TronSolver(nlp; μ₀ = μ₀, μ₁ = μ₁, σ = σ, subsolver_kwargs...)
  for k in subsolver_keys
    pop!(dict, k)
  end
  return solve!(solver, nlp; x = x, dict...)
end

function SolverCore.solve!(
  solver::TronSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Float64 = 30.0,
  max_cgiter::Int = 50,
  use_only_objgrad::Bool = false,
  cgtol::T = T(0.1),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
) where {T, V <: AbstractVector{T}}
  if !(nlp.meta.minimize)
    error("tron only works for minimization problem")
  end
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error("tron should only be called for unconstrained or bound-constrained problems")
  end

  # parameters
  μ₀ = value(solver.params.μ₀)
  μ₁ = value(solver.params.μ₁)
  σ = value(solver.params.σ)

  reset!(stats)
  ℓ = nlp.meta.lvar
  u = nlp.meta.uvar
  n = nlp.meta.nvar

  if (verbose > 0 && !(u ≥ x ≥ ℓ))
    @warn "Warning: Initial guess is not within bounds."
  end

  start_time = time()
  set_time!(stats, 0.0)

  solver.x .= x
  x = solver.x
  xc = solver.xc
  gx = solver.gx
  gt = solver.gt
  gn = solver.gn
  gpx = solver.gpx
  s = solver.s
  Hs = solver.Hs
  H = solver.H

  x .= max.(ℓ, min.(x, u))
  fx, _ = objgrad!(nlp, x, gx)
  # gt = use_only_objgrad ? zeros(T, n) : T[]
  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -one(T))
  πx = nrm2(n, gpx)
  ϵ = atol + rtol * πx
  fmin = min(-one(T), fx) / eps(T)
  optimal = πx <= ϵ
  unbounded = fx < fmin

  set_iter!(stats, 0)
  set_objective!(stats, fx)
  set_dual_residual!(stats, πx)

  if isa(nlp, QuasiNewtonModel)
    gn .= gx
  end

  αC = one(T)
  tr = solver.tr
  tr.radius = tr.initial_radius = min(max(one(T), πx / 10), tr.max_radius)
  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :cgstatus],
    [Int, T, T, T, T, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )
  verbose > 0 && @info log_row([stats.iter, fx, πx, T, T, String])

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
    # Current iteration
    xc .= x # implicitly update H = hess_op!(nlp, xc, Hs)
    fc = fx
    Δ = tr.radius

    αC, s, cauchy_status = cauchy!(x, H, gx, Δ, αC, ℓ, u, s, Hs, μ₀ = μ₀, μ₁ = μ₁, σ = σ)

    if cauchy_status != :success
      stats.status = cauchy_status
      done = true
      continue
    end

    cginfo = projected_newton!(
      solver,
      x,
      H,
      gx,
      Δ,
      cgtol,
      ℓ,
      u,
      s,
      Hs,
      max_cgiter = max_cgiter,
      max_time = max_time - stats.elapsed_time,
      subsolver_verbose = subsolver_verbose,
    )

    slope = dot(n, gx, s)
    qs = dot(n, s, Hs) / 2 + slope
    fx = if use_only_objgrad
      objgrad!(nlp, x, gt)[1]
    else
      obj(nlp, x)
    end

    ared, pred = aredpred!(tr, nlp, fc, fx, qs, x, s, slope)
    if pred ≥ 0
      stats.status = :neg_pred
      done = true
      continue
    end
    tr.ratio = ared / pred

    if acceptable(tr)
      num_success_iters += 1
      if use_only_objgrad
        gx .= gt
      else
        grad!(nlp, x, gx)
      end
      project_step!(gpx, x, gx, ℓ, u, -one(T))
      πx = nrm2(n, gpx)

      if isa(nlp, QuasiNewtonModel)
        gn .-= gx
        gn .*= -1  # gn = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, gn)
        gn .= gx
      end
    end

    if !acceptable(tr)
      fx = fc
      x .= xc
    end

    set_iter!(stats, stats.iter + 1)

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row([stats.iter, fx, πx, Δ, tr.ratio, cginfo])

    s_norm = nrm2(n, s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    optimal = πx <= ϵ
    unbounded = fx < fmin

    set_objective!(stats, fx)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, πx)

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
  verbose > 0 && @info log_row(Any[stats.iter, fx, πx, tr.radius])

  set_solution!(stats, x)
  stats
end

"""
    s = projected_line_search!(x, H, g, d, ℓ, u, Hs, μ₀)

Performs a projected line search, searching for a step size `t` such that

    0.5sᵀHs + sᵀg ≦ μ₀sᵀg,

where `s = P(x + t * d) - x`, while remaining on the same face as `x + d`.
Backtracking is performed from t = 1.0. `x` is updated in place.
"""
function projected_line_search!(
  x::AbstractVector{T},
  H::Union{AbstractMatrix, AbstractLinearOperator},
  g::AbstractVector{T},
  d::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  Hs::AbstractVector{T},
  s::AbstractVector{T},
  μ₀::Real,
) where {T <: Real}
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0

  s .= zero(T)
  Hs .= zero(T)

  search = true
  while search && α > brkmin
    nsteps += 1
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    if qs <= μ₀ * slope
      search = false
    else
      α /= 2
    end
  end
  if α < 1 && α < brkmin
    α = brkmin
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
  end

  project_step!(s, x, d, ℓ, u, α)
  x .= x .+ s

  return s
end

"""`α, s = cauchy!(x, H, g, Δ, ℓ, u, s, Hs; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ μ₀sᵀg.
"""
function cauchy!(
  x::AbstractVector{T},
  H::Union{AbstractMatrix, AbstractLinearOperator},
  g::AbstractVector{T},
  Δ::Real,
  α::Real,
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  s::AbstractVector{T},
  Hs::AbstractVector{T};
  μ₀::Real = T(TRON_μ₀),
  μ₁::Real = T(TRON_μ₁),
  σ::Real = T(TRON_σ),
) where {T <: Real}
  # TODO: Use brkmin to care for g direction
  s .= .-g
  _, _, brkmax = breakpoints(x, s, ℓ, u)

  n = length(x)
  s .= zero(T)
  Hs .= zero(T)

  status = :success

  project_step!(s, x, g, ℓ, u, -α)

  # Interpolate or extrapolate
  s_norm = nrm2(n, s)
  if s_norm > μ₁ * Δ
    interp = true
  else
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    interp = qs >= μ₀ * slope
  end

  if interp
    search = true
    while search
      α /= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = nrm2(n, s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        search = qs >= μ₀ * slope
      end
      # TODO: Correctly assess why this fails
      if α < sqrt(nextfloat(zero(α)))
        search = false
        status = :small_step
      end
    end
  else
    search = true
    αs = α
    while search && α <= brkmax
      α *= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = nrm2(n, s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        if qs <= μ₀ * slope
          αs = α
        end
      else
        search = false
      end
    end
    # Recover the last successful step
    α = αs
    s = project_step!(s, x, g, ℓ, u, -α)
  end
  return α, s, status
end

"""

    projected_newton!(solver, x, H, g, Δ, cgtol, ℓ, u, s, Hs; max_time = Inf, max_cgiter = 50, subsolver_verbose = 0)

Compute an approximate solution `d` for

min q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_newton!(
  solver::TronSolver{T},
  x::AbstractVector{T},
  H::Union{AbstractMatrix, AbstractLinearOperator},
  g::AbstractVector{T},
  Δ::T,
  cgtol::T,
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  s::AbstractVector{T},
  Hs::AbstractVector{T};
  max_cgiter::Int = 50,
  max_time::Float64 = Inf,
  subsolver_verbose = 0,
) where {T <: Real}
  start_time, elapsed_time = time(), 0.0
  n = length(x)
  status = ""

  cg_solver, cgs_rhs = solver.cg_solver, solver.cg_rhs
  cg_op_diag, ZHZ = solver.cg_op_diag, solver.ZHZ
  w = solver.temp
  ifix = solver.ifix

  mul!(Hs, H, s)

  # Projected Newton Step
  exit_optimal, exit_pcg, exit_itmax, exit_time = false, false, false, false
  iters = 0
  x .= x .+ s
  project!(x, x, ℓ, u)
  while !(exit_optimal || exit_pcg || exit_itmax || exit_time)
    active!(ifix, x, ℓ, u)
    if sum(ifix) == n
      exit_optimal = true
      continue
    end

    gfnorm = 0
    for i = 1:n
      cgs_rhs[i] = ifix[i] ? 0 : -g[i]
      gfnorm += cgs_rhs[i]^2
      cgs_rhs[i] -= ifix[i] ? 0 : Hs[i]
      cg_op_diag[i] = ifix[i] ? 0 : 1 # implictly changes cg_op and so ZHZ
    end

    Krylov.cg!(
      cg_solver,
      ZHZ,
      cgs_rhs,
      radius = Δ,
      rtol = cgtol,
      atol = zero(T),
      timemax = max_time - elapsed_time,
      verbose = subsolver_verbose,
    )

    st, stats = cg_solver.x, cg_solver.stats
    status = stats.status
    iters += 1

    # Projected line search
    cgs_rhs .*= -1
    projected_line_search!(x, ZHZ, cgs_rhs, st, ℓ, u, Hs, w, value(solver.params.μ₀))
    s .+= w

    mul!(Hs, H, s)

    newnorm = 0
    for i = 1:n
      cgs_rhs[i] = ifix[i] ? 0 : Hs[i] + g[i]
      newnorm += cgs_rhs[i]^2
    end

    if √newnorm <= cgtol * √gfnorm
      exit_optimal = true
    elseif status == "on trust-region boundary"
      exit_pcg = true
    elseif iters >= max_cgiter
      exit_itmax = true
    end

    elapsed_time = time() - start_time
    exit_time = elapsed_time >= max_time
  end

  status = if exit_optimal
    "stationary point found"
  elseif exit_pcg
    "on trust-region boundary"
  elseif exit_itmax
    "maximum number of iterations"
  elseif exit_time
    "time limit exceeded"
  else
    status # unknown
  end

  return status
end
