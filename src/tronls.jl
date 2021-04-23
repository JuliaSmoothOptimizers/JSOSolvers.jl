const tronls_allowed_subsolvers = [:cgls, :crls, :lsqr, :lsmr]

tron(nls::AbstractNLSModel; variant = :GaussNewton, kwargs...) = tron(Val(variant), nls; kwargs...)

"""
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)

Compute `slope = dot(As, Fx)` and `qs = dot(As, As) / 2 + slope`. Use `As` to store `A * s`.
"""
function compute_As_slope_qs!(
  As::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  s::AbstractVector{T},
  Fx::AbstractVector{T},
) where {T <: Real}
  As .= A * s
  slope = dot(As, Fx)
  qs = dot(As, As) / 2 + slope
  return slope, qs
end

"""
    tron(nls)

A pure Julia implementation of a trust-region solver for bound-constrained
nonlinear least squares optimization:

    min ½‖F(x)‖²    s.t.    ℓ ≦ x ≦ u

This is an adaptation of the TRON method described in

Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
"""
function tron(
  ::Val{:GaussNewton},
  nlp::AbstractNLSModel;
  subsolver_logger::AbstractLogger = NullLogger(),
  x::AbstractVector = copy(nlp.meta.x0),
  subsolver::Symbol = :lsmr,
  μ₀::Real = eltype(x)(1e-2),
  μ₁::Real = one(eltype(x)),
  σ::Real = eltype(x)(10),
  max_eval::Int = -1,
  max_time::Real = 30.0,
  max_cgiter::Int = 50,
  cgtol::Real = eltype(x)(0.1),
  atol::Real = √eps(eltype(x)),
  rtol::Real = √eps(eltype(x)),
  fatol::Real = zero(eltype(x)),
  frtol::Real = eps(eltype(x))^eltype(x)(2 / 3),
)
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error("tron should only be called for unconstrained or bound-constrained problems")
  end

  T = eltype(x)
  ℓ = T.(nlp.meta.lvar)
  u = T.(nlp.meta.uvar)
  n = nlp.meta.nvar
  m = nlp.nls_meta.nequ

  iter = 0
  start_time = time()
  el_time = 0.0

  # Preallocation
  Av = zeros(T, m)
  Atv = zeros(T, n)
  gpx = zeros(T, n)
  xc = zeros(T, n)
  Hs = zeros(T, n)

  F(x) = residual(nlp, x)
  A(x) = jac_op_residual!(nlp, x, Av, Atv)

  x .= max.(ℓ, min.(x, u))
  Fx = F(x)
  fx = dot(Fx, Fx) / 2
  Ax = A(x)
  gx = copy(Ax' * Fx)
  Fc = zeros(T, m)
  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -one(T))
  πx = nrm2(n, gpx)
  ϵ = atol + rtol * πx
  fmin = min(-one(T), fx) / eps(eltype(x))
  optimal = πx <= ϵ
  tired = el_time > max_time || neval_obj(nlp) > max_eval ≥ 0
  unbounded = fx < fmin
  stalled = false
  status = :unknown

  αC = one(T)
  tr = TRONTrustRegion(min(max(one(T), πx / 10), 100))
  @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :cgstatus],
    [Int, T, T, T, T, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )
  while !(optimal || tired || stalled || unbounded)
    # Current iteration
    xc .= x
    fc = fx
    Fc .= Fx
    Δ = get_property(tr, :radius)

    αC, s, cauchy_status = cauchy_ls(x, Ax, Fx, gx, Δ, αC, ℓ, u, μ₀ = μ₀, μ₁ = μ₁, σ = σ)

    if cauchy_status != :success
      @error "Cauchy step returned: $cauchy_status"
      status = cauchy_status
      stalled = true
      continue
    end
    s, As, cgits, cginfo = with_logger(subsolver_logger) do
      projected_gauss_newton!(
        x,
        Ax,
        Fx,
        Δ,
        cgtol,
        s,
        ℓ,
        u,
        subsolver = subsolver,
        max_cgiter = max_cgiter,
      )
    end
    slope = dot(m, Fx, As)
    qs = dot(As, As) / 2 + slope
    Fx = F(x)
    fx = dot(Fx, Fx) / 2

    ared, pred, quad_min = aredpred(tr, nlp, fc, fx, qs, x, s, slope)
    if pred ≥ 0
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred
    tr.quad_min = quad_min
    @info log_row([iter, fx, πx, Δ, tr.ratio, cginfo])

    s_norm = nrm2(n, s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    if acceptable(tr)
      num_success_iters += 1
      Ax = A(x)
      gx .= Ax' * Fx
      project_step!(gpx, x, gx, ℓ, u, -one(T))
      πx = nrm2(n, gpx)
    end

    # No post-iteration

    if !acceptable(tr)
      Fx .= Fc
      fx = fc
      x .= xc
    end

    iter += 1
    el_time = time() - start_time
    tired = el_time > max_time || neval_obj(nlp) > max_eval ≥ 0
    optimal = πx <= ϵ
    unbounded = fx < fmin
  end
  @info log_row(Any[iter, fx, πx, get_property(tr, :radius)])

  if tired
    if el_time > max_time
      status = :max_time
    elseif neval_obj(nlp) > max_eval ≥ 0
      status = :max_eval
    end
  elseif optimal
    status = :first_order
  elseif unbounded
    status = :unbounded
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = fx,
    dual_feas = πx,
    iter = iter,
    elapsed_time = el_time,
  )
end

"""`s = projected_line_search_ls!(x, A, g, d, ℓ, u; μ₀ = 1e-2)`

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
  u::AbstractVector{T};
  μ₀::Real = T(1e-2),
) where {T <: Real}
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0
  n = length(x)
  m = length(Fx)

  s = zeros(T, n)
  As = zeros(T, m)

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

"""`α, s = cauchy_ls(x, A, Fx, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ½‖As + Fx‖² - ½‖Fx‖²     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ μ₀gᵀs,

where g = AᵀFx.
"""
function cauchy_ls(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  g::AbstractVector{T},
  Δ::Real,
  α::Real,
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  μ₀::Real = T(1e-2),
  μ₁::Real = one(T),
  σ::Real = T(10),
) where {T <: Real}
  # TODO: Use brkmin to care for g direction
  _, _, brkmax = breakpoints(x, -g, ℓ, u)
  n = length(x)
  m = length(Fx)
  s = zeros(T, n)
  As = zeros(T, m)

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
        stalled = true
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

"""`projected_gauss_newton!(x, A, Fx, Δ, gctol, s, max_cgiter, ℓ, u)`

Compute an approximate solution `d` for

    min q(d) = ½‖Ad + Fx‖² - ½‖Fx‖²     s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_gauss_newton!(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  Δ::Real,
  cgtol::Real,
  s::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  subsolver::Symbol = :lsmr,
  max_cgiter::Int = 50,
) where {T <: Real}
  n = length(x)
  status = ""
  subsolver in tronls_allowed_subsolvers ||
    error("subproblem solver must be one of $tronls_allowed_subsolvers")
  lssolver = eval(subsolver)

  As = A * s

  # Projected Newton Step
  exit_optimal = false
  exit_pcg = false
  exit_itmax = false
  iters = 0
  xt = x .+ s
  project!(xt, xt, ℓ, u)
  while !(exit_optimal || exit_pcg || exit_itmax)
    ifree = setdiff(1:n, active(xt, ℓ, u))
    if length(ifree) == 0
      exit_optimal = true
      continue
    end
    Z = opExtension(ifree, n)
    @views wa = Fx
    @views Ffree = As + wa
    wanorm = norm(wa)

    AZ = A * Z
    st, stats = lssolver(AZ, -Ffree, radius = Δ, rtol = cgtol, atol = zero(T))
    iters += 1
    status = stats.status

    # Projected line search
    xfree = @view xt[ifree]
    @views w = projected_line_search_ls!(xfree, AZ, Ffree, st, ℓ[ifree], u[ifree])
    @views s[ifree] .+= w

    As .= A * s

    @views Ffree .= As .+ Fx
    if norm(Z' * A' * Ffree) <= cgtol * wanorm
      exit_optimal = true
    elseif status == "on trust-region boundary"
      exit_pcg = true
    elseif iters >= max_cgiter
      exit_itmax = true
    end
  end
  status = if exit_optimal
    "stationary point found"
  elseif exit_itmax
    "maximum number of iterations"
  else
    status # on trust-region
  end

  x .= xt

  return s, As, iters, status
end
