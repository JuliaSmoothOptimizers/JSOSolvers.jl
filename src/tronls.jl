export TronSolverNLS, TRONLSParameterSet

const tronls_allowed_subsolvers = [CglsSolver, CrlsSolver, LsqrSolver, LsmrSolver]

tron(nls::AbstractNLSModel; variant = :GaussNewton, kwargs...) = tron(Val(variant), nls; kwargs...)

# Default algorithm parameter values
const TRONLS_μ₀ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1 // 100), "T(1 / 100)")
const TRONLS_μ₁ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1), "T(1)")
const TRONLS_σ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(10), "T(10)")

"""
    TRONLSParameterSet{T} <: AbstractParameterSet

This structure designed for `tron` regroups the following parameters:
  - `μ₀`: algorithm parameter in (0, 0.5).
  - `μ₁`: algorithm parameter in (0, +∞).
  - `σ`: algorithm parameter in (1, +∞).

An additional constructor is

    TRONLSParameterSet(nlp: kwargs...)

where the kwargs are the parameters above.

Default values are:
  - `μ₀::T = $(TRONLS_μ₀)`
  - `μ₁::T = $(TRONLS_μ₁)`
  - `σ::T = $(TRONLS_σ)`
"""
struct TRONLSParameterSet{T} <: AbstractParameterSet
  μ₀::Parameter{T, RealInterval{T}}
  μ₁::Parameter{T, RealInterval{T}}
  σ::Parameter{T, RealInterval{T}}
end

# add a default constructor
function TRONLSParameterSet(
  nlp::AbstractNLPModel{T};
  μ₀::T = get(TRONLS_μ₀, nlp),
  μ₁::T = get(TRONLS_μ₁, nlp),
  σ::T = get(TRONLS_σ, nlp),
) where {T}
  TRONLSParameterSet(
    Parameter(μ₀, RealInterval(T(0), T(1 // 2), lower_open = true)),
    Parameter(μ₁, RealInterval(T(0), T(Inf), lower_open = true)),
    Parameter(σ, RealInterval(T(1), T(Inf), lower_open = true)),
  )
end

"""
    tron(nls; kwargs...)

A pure Julia implementation of a trust-region solver for bound-constrained
nonlinear least-squares problems:

    min ½‖F(x)‖²    s.t.    ℓ ≦ x ≦ u

For advanced usage, first define a `TronSolverNLS` to preallocate the memory used in the algorithm, and then call `solve!`:
    solver = TronSolverNLS(nls, subsolver_type::Type{<:KrylovSolver} = LsmrSolver; kwargs...)
    solve!(solver, nls; kwargs...)

# Arguments
- `nls::AbstractNLSModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `x::V = nlp.meta.x0`: the initial guess.
- `subsolver_type::Symbol = LsmrSolver`: `Krylov.jl` method used as subproblem solver, see `JSOSolvers.tronls_allowed_subsolvers` for a list.
- `μ₀::T = $(TRONLS_μ₀)`: algorithm parameter, see [`TRONLSParameterSet`](@ref).
- `μ₁::T = $(TRONLS_μ₁)`: algorithm parameter, see [`TRONLSParameterSet`](@ref).
- `σ::T = $(TRONLS_σ)`: algorithm parameter, see [`TRONLSParameterSet`](@ref).
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `max_cgiter::Int = 50`: subproblem iteration limit.
- `cgtol::T = T(0.1)`: subproblem tolerance.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖x - Proj(x - ∇f(xᵏ))‖ ≤ atol + rtol * ‖∇f(x⁰)‖. Proj denotes here the projection over the bounds.
- `Fatol::T = √eps(T)`: absolute tolerance on the residual.
- `Frtol::T = eps(T)`: relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver.

The keyword arguments of `TronSolverNLS` are passed to the [`TRONTrustRegion`](https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/main/src/trust-region/tron-trust-region.jl) constructor.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(Callback_docstring)

# References
This is an adaptation for bound-constrained nonlinear least-squares problems of the TRON method described in

    Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
    Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
    DOI: 10.1137/S1052623498345075

# Examples
```jldoctest; output = false
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))
stats = tron(nls)
# output
"Execution stats: first-order stationary"
```

```jldoctest; output = false
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))
solver = TronSolverNLS(nls)
stats = solve!(solver, nls)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct TronSolverNLS{
  T,
  V <: AbstractVector{T},
  Sub <: KrylovSolver{T, T, V},
  Op <: AbstractLinearOperator{T},
  Aop <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
  x::V
  xc::V
  temp::V
  gx::V
  gt::V
  gpx::V
  s::V
  tr::TRONTrustRegion{T, V}
  Fx::V
  Fc::V
  Av::V
  Atv::V
  A::Op
  As::V

  ifix::BitVector

  ls_rhs::V
  ls_op_diag::V
  ls_op::LinearOperator{T}

  AZ::Aop
  ls_subsolver::Sub
  params::TRONLSParameterSet{T}
end

function TronSolverNLS(
  nlp::AbstractNLSModel{T, V};
  μ₀::T = get(TRONLS_μ₀, nlp),
  μ₁::T = get(TRONLS_μ₁, nlp),
  σ::T = get(TRONLS_σ, nlp),
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
  max_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)),
  kwargs...,
) where {T, V <: AbstractVector{T}}
  subsolver_type in tronls_allowed_subsolvers ||
    error("subproblem solver must be one of $(tronls_allowed_subsolvers)")

  params = TRONLSParameterSet(nlp; μ₀ = μ₀, μ₁ = μ₁, σ = σ)
  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ
  x = V(undef, nvar)
  xc = V(undef, nvar)
  temp = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gpx = V(undef, nvar)
  s = V(undef, nvar)
  tr = TRONTrustRegion(gt, min(one(T), max_radius - eps(T)); max_radius = max_radius, kwargs...)

  Fx = V(undef, nequ)
  Fc = V(undef, nequ)

  Av = V(undef, nequ)
  Atv = V(undef, nvar)
  A = jac_op_residual!(nlp, xc, Av, Atv)
  As = V(undef, nequ)

  ifix = BitVector(undef, nvar)

  ls_rhs = V(undef, nequ)
  ls_op_diag = V(undef, nvar)
  ls_op = opDiagonal(ls_op_diag)

  AZ = A * ls_op
  ls_subsolver = subsolver_type(AZ, As)
  Sub = typeof(ls_subsolver)

  return TronSolverNLS{T, V, Sub, typeof(A), typeof(AZ)}(
    x,
    xc,
    temp,
    gx,
    gt,
    gpx,
    s,
    tr,
    Fx,
    Fc,
    Av,
    Atv,
    A,
    As,
    ifix,
    ls_rhs,
    ls_op_diag,
    ls_op,
    AZ,
    ls_subsolver,
    params,
  )
end

function SolverCore.reset!(solver::TronSolverNLS)
  solver.tr.good_grad = false
  solver
end
function SolverCore.reset!(solver::TronSolverNLS, nlp::AbstractNLPModel)
  solver.A = jac_op_residual!(nlp, solver.xc, solver.Av, solver.Atv)
  solver.AZ = solver.A * solver.ls_op
  reset!(solver)
  solver
end

@doc (@doc TronSolverNLS) function tron(
  ::Val{:GaussNewton},
  nlp::AbstractNLSModel{T, V};
  x::V = nlp.meta.x0,
  μ₀::Real = get(TRONLS_μ₀, nlp),
  μ₁::Real = get(TRONLS_μ₁, nlp),
  σ::Real = get(TRONLS_σ, nlp),
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
  kwargs...,
) where {T, V}
  dict = Dict(kwargs)
  subsolver_keys = intersect(keys(dict), tron_keys)
  subsolver_kwargs = Dict(k => dict[k] for k in subsolver_keys)
  solver = TronSolverNLS(
    nlp,
    μ₀ = μ₀,
    μ₁ = μ₁,
    σ = σ,
    subsolver_type = subsolver_type;
    subsolver_kwargs...,
  )
  for k in subsolver_keys
    pop!(dict, k)
  end
  return solve!(solver, nlp; x = x, dict...)
end

function SolverCore.solve!(
  solver::TronSolverNLS{T, V},
  nlp::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Real = 30.0,
  max_cgiter::Int = 50,
  cgtol::T = T(0.1),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::T = √eps(T),
  Frtol::T = eps(T),
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
  m = nlp.nls_meta.nequ

  if (verbose > 0 && !(u ≥ x ≥ ℓ))
    @warn "Warning: Initial guess is not within bounds."
  end

  start_time = time()
  set_time!(stats, 0.0)

  solver.x .= x
  x = solver.x
  xc = solver.xc
  gx = solver.gx

  # Preallocation
  s = solver.s
  As = solver.As
  Ax = solver.A

  gpx = solver.gpx
  Fx, Fc = solver.Fc, solver.Fx

  x .= max.(ℓ, min.(x, u))
  residual!(nlp, x, Fx)
  fx, gx = objgrad!(nlp, x, gx, Fx, recompute = false)
  gt = solver.gt

  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -one(T))
  πx = nrm2(n, gpx)
  ϵ = atol + rtol * πx
  fmin = min(-one(T), fx) / eps(T)
  optimal = πx <= ϵ
  unbounded = fx < fmin
  ϵF = Fatol + Frtol * 2 * √fx
  project_step!(gpx, x, x, ℓ, u, zero(T)) # Proj(x) - x
  small_residual = (norm(gpx) <= ϵ) && (2 * √fx <= ϵF)

  set_iter!(stats, 0)
  set_objective!(stats, fx)
  set_dual_residual!(stats, πx)

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
      small_residual = small_residual,
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
    xc .= x # implicitly update Ax
    fc = fx
    Fc .= Fx
    Δ = tr.radius

    αC, s, cauchy_status = cauchy_ls!(x, Ax, Fx, gx, Δ, αC, ℓ, u, s, As, μ₀ = μ₀, μ₁ = μ₁, σ = σ)

    if cauchy_status != :success
      @error "Cauchy step returned: $cauchy_status"
      stats.status = cauchy_status
      done = true
      continue
    end
    cginfo = projected_gauss_newton!(
      solver,
      x,
      Ax,
      Fx,
      Δ,
      cgtol,
      s,
      ℓ,
      u,
      As,
      max_cgiter = max_cgiter,
      max_time = max_time - stats.elapsed_time,
      subsolver_verbose = subsolver_verbose,
    )

    slope = dot(m, Fx, As)
    qs = dot(As, As) / 2 + slope
    residual!(nlp, x, Fx)
    fx = obj(nlp, x, Fx, recompute = false)

    ared, pred = aredpred!(tr, nlp, Fx, fc, fx, qs, x, s, slope)
    if pred ≥ 0
      stats.status = :neg_pred
      done = true
      continue
    end
    tr.ratio = ared / pred

    s_norm = nrm2(n, s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    if acceptable(tr)
      num_success_iters += 1
      if tr.good_grad
        gx .= tr.gt
        tr.good_grad = false
      else
        grad!(nlp, x, gx, Fx, recompute = false)
      end
      project_step!(gpx, x, gx, ℓ, u, -one(T))
      πx = nrm2(n, gpx)
    end

    # No post-iteration

    if !acceptable(tr)
      Fx .= Fc
      fx = fc
      x .= xc
    end

    set_iter!(stats, stats.iter + 1)
    set_objective!(stats, fx)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, πx)

    optimal = πx <= ϵ
    project_step!(gpx, x, x, ℓ, u, zero(T)) # Proj(x) - x
    small_residual = (norm(gpx) <= ϵ) && (2 * √fx <= ϵF)
    unbounded = fx < fmin

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row([stats.iter, fx, πx, Δ, tr.ratio, cginfo])

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        small_residual = small_residual,
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

"""`s = projected_line_search_ls!(x, A, g, d, ℓ, u, As, s; μ₀ = 1e-2)`

Performs a projected line search, searching for a step size `t` such that

    ½‖As + Fx‖² ≤ ½‖Fx‖² + μ₀FxᵀAs

where `s = P(x + t * d) - x`, while remaining on the same face as `x + d`.
Backtracking is performed from t = 1.0. `x` is updated in place.
"""
function projected_line_search_ls!(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  d::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  As::AbstractVector{T},
  s::AbstractVector{T};
  μ₀::Real = T(1e-2),
) where {T <: Real}
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0
  n = length(x)
  m = length(Fx)

  s .= zero(T)
  As .= zero(T)

  search = true
  while search && α > brkmin
    nsteps += 1
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
    if qs <= μ₀ * slope
      search = false
    else
      α /= 2
    end
  end
  if α < 1 && α < brkmin
    α = brkmin
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
  end

  project_step!(s, x, d, ℓ, u, α)
  x .= x .+ s

  return s
end

"""`α, s = cauchy_ls!(x, A, Fx, g, Δ, ℓ, u, s, As; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ½‖As + Fx‖² - ½‖Fx‖²     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ μ₀gᵀs,

where g = AᵀFx.
"""
function cauchy_ls!(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  g::AbstractVector{T},
  Δ::Real,
  α::Real,
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  s::AbstractVector{T},
  As::AbstractVector{T};
  μ₀::Real = T(1e-2),
  μ₁::Real = one(T),
  σ::Real = T(10),
) where {T <: Real}
  # TODO: Use brkmin to care for g direction
  s .= .-g
  _, _, brkmax = breakpoints(x, s, ℓ, u)

  n = length(x)
  s .= zero(T)
  As .= zero(T)

  status = :success

  project_step!(s, x, g, ℓ, u, -α)

  # Interpolate or extrapolate
  s_norm = nrm2(n, s)
  if s_norm > μ₁ * Δ
    interp = true
  else
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
    interp = qs >= μ₀ * slope
  end

  if interp
    search = true
    while search
      α /= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = nrm2(n, s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_As_slope_qs!(As, A, s, Fx)
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
        slope, qs = compute_As_slope_qs!(As, A, s, Fx)
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

    projected_gauss_newton!(solver, x, A, Fx, Δ, gctol, s, max_cgiter, ℓ, u; max_cgiter = 50, max_time = Inf, subsolver_verbose = 0)

Compute an approximate solution `d` for

    min q(d) = ½‖Ad + Fx‖² - ½‖Fx‖²     s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_gauss_newton!(
  solver::TronSolverNLS{T},
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  Δ::T,
  cgtol::T,
  s::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  As::AbstractVector{T};
  max_cgiter::Int = 50,
  max_time::Float64 = Inf,
  subsolver_verbose = 0,
) where {T <: Real}
  start_time, elapsed_time = time(), 0.0
  n = length(x)
  status = ""

  w = solver.temp
  ls_rhs = solver.ls_rhs
  ls_op_diag, AZ = solver.ls_op_diag, solver.AZ

  ifix = solver.ifix
  ls_subsolver = solver.ls_subsolver

  mul!(As, A, s)

  Fxnorm = norm(Fx)

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

    for i = 1:n
      ls_op_diag[i] = ifix[i] ? 0 : 1 # implictly changes ls_op and so AZ
    end

    ls_rhs .= .-As .- Fx
    Krylov.solve!(
      ls_subsolver,
      AZ,
      ls_rhs,
      radius = Δ,
      rtol = cgtol,
      atol = zero(T),
      timemax = max_time - elapsed_time,
      verbose = subsolver_verbose,
    )

    st, stats = ls_subsolver.x, ls_subsolver.stats
    iters += 1
    status = stats.status

    # Projected line search
    ls_rhs .*= -1
    projected_line_search_ls!(x, AZ, ls_rhs, st, ℓ, u, As, w)
    s .+= w

    mul!(As, A, s)

    ls_rhs .= .-As .- Fx
    mul!(w, AZ', ls_rhs)
    if norm(w) <= cgtol * Fxnorm
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
