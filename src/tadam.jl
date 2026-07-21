export tadam, TADAMSolver, TADAMParameterSet

"""
    TADAMParameterSet([T=Float64]; η1, η2, γ1, γ2, γ3, β1, β2, ϵ_v, θ1, θ2, Δmax)

Parameter set for the TADAM solver. Controls algorithmic tolerances, momentum parameters, and step acceptance.

# Keyword Arguments
- `η1 = eps(T)^(1/4)`: Accept step if actual/predicted reduction ≥ η1.
- `η2 = T(0.95)`: Step is very successful if reduction ≥ η2.
- `γ1 = T(0.5)`: Radius decrease factor on rejected step.
- `γ2 = T(2.0)`: Radius increase factor on very successful step.
- `γ3 = T(0.5)`: Momentum contribution decrease factor on rejected step.
- `β1 = T(0.9)`: Constant in the momentum term.
- `β2 = T(0.999)`: Constant in the RMSProp term.
- `ϵ_v = T(1e-8)`: RMSProp epsilon to prevent division by zero.
- `θ1 = T(0.1)`: Momentum contribution parameter for convergence condition 1.
- `θ2 = T(1.0)`: Momentum contribution parameter for convergence condition 2.
- `Δmax = 1/eps(T)`: Maximum step parameter for trust-region.
"""
struct TADAMParameterSet{T} <: AbstractParameterSet
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  γ3::Parameter{T, RealInterval{T}}
  β1::Parameter{T, RealInterval{T}}
  β2::Parameter{T, RealInterval{T}}
  ϵ_v::Parameter{T, RealInterval{T}}
  θ1::Parameter{T, RealInterval{T}}
  θ2::Parameter{T, RealInterval{T}}
  Δmax::Parameter{T, RealInterval{T}}
end

# Default parameter values
const TADAM_η1 = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))^(T(1) / T(4))
end, "eps(T)^(1/4)")
const TADAM_η2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.95), "T(0.95)")
const TADAM_γ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const TADAM_γ2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(2.0), "T(2.0)")
const TADAM_γ3 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const TADAM_β1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.9), "T(0.9)")
const TADAM_β2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.999), "T(0.999)")
const TADAM_ϵ_v = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1e-8), "T(1e-8)")
const TADAM_θ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.1), "T(0.1)")
const TADAM_θ2 = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  inv(T(eps(T))^(T(1) / T(3))) # Inverted to create a macroscopic bound
end, "1/eps(T)^(1/3)")
const TADAM_Δmax = DefaultParameter(nlp -> inv(eps(eltype(nlp.meta.x0))), "1/eps(T)")

function TADAMParameterSet(
  nlp::AbstractNLPModel;
  η1::T = get(TADAM_η1, nlp),
  η2::T = get(TADAM_η2, nlp),
  γ1::T = get(TADAM_γ1, nlp),
  γ2::T = get(TADAM_γ2, nlp),
  γ3::T = get(TADAM_γ3, nlp),
  β1::T = get(TADAM_β1, nlp),
  β2::T = get(TADAM_β2, nlp),
  ϵ_v::T = get(TADAM_ϵ_v, nlp),
  θ1::T = get(TADAM_θ1, nlp),
  θ2::T = get(TADAM_θ2, nlp),
  Δmax::T = get(TADAM_Δmax, nlp),
) where {T}
  @assert zero(T) < η1 ≤ η2 < one(T) "η1, η2 must satisfy 0 < η1 ≤ η2 < 1"
  @assert zero(T) < γ1 < one(T) "γ1 must satisfy 0 < γ1 < 1"
  @assert γ2 > one(T) "γ2 must satisfy γ2 > 1"
  @assert zero(T) < γ3 < one(T) "γ3 must satisfy 0 < γ3 < 1"
  @assert zero(T) ≤ β1 < one(T) "β1 must satisfy 0 ≤ β1 < 1"
  @assert zero(T) ≤ β2 < one(T) "β2 must satisfy 0 ≤ β2 < 1"
  @assert zero(T) < ϵ_v "ϵ_v must be strictly positive"
  @assert zero(T) < θ1 < one(T) "θ1 must satisfy 0 < θ1 < 1"
  @assert θ2 > zero(T) "θ2 must be strictly positive"
  @assert Δmax > zero(T) "Δmax must be strictly positive"

  TADAMParameterSet{T}(
    Parameter(η1, RealInterval(zero(T), one(T), lower_open = false, upper_open = true)),
    Parameter(η2, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(γ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(γ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ3, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(β1, RealInterval(zero(T), one(T), lower_open = false, upper_open = true)),
    Parameter(β2, RealInterval(zero(T), one(T), lower_open = false, upper_open = true)),
    Parameter(ϵ_v, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(θ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(θ2, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(Δmax, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
  )
end

"""
    tadam(nlp; kwargs...)

Trust-region embedded ADAM (TADAM) algorithm for unconstrained optimization. 

For advanced usage, first define a `TADAMSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = TADAMSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.
"""

mutable struct TADAMSolver{T, V, M <: AbstractNLPModel{T, V}} <: AbstractOptimizationSolver
  x::V
  xt::V
  gx::V
  m::V
  d::V
  v::V
  s::V
  sc::V        # Cauchy step
  s_infty::V   # Subproblem step
  p::V
  Δ::T
  step_accepted::Bool
  params::TADAMParameterSet{T}
end

function TADAMSolver(
  nlp::AbstractNLPModel{T, V};
  η1::T = get(TADAM_η1, nlp),
  η2::T = get(TADAM_η2, nlp),
  γ1::T = get(TADAM_γ1, nlp),
  γ2::T = get(TADAM_γ2, nlp),
  γ3::T = get(TADAM_γ3, nlp),
  β1::T = get(TADAM_β1, nlp),
  β2::T = get(TADAM_β2, nlp),
  ϵ_v::T = get(TADAM_ϵ_v, nlp),
  θ1::T = get(TADAM_θ1, nlp),
  θ2::T = get(TADAM_θ2, nlp),
  Δmax::T = get(TADAM_Δmax, nlp),
) where {T, V}
  params = TADAMParameterSet(
    nlp;
    η1 = η1,
    η2 = η2,
    γ1 = γ1,
    γ2 = γ2,
    γ3 = γ3,
    β1 = β1,
    β2 = β2,
    ϵ_v = ϵ_v,
    θ1 = θ1,
    θ2 = θ2,
    Δmax = Δmax,
  )

  x = similar(nlp.meta.x0)
  xt = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  m = fill!(similar(nlp.meta.x0), zero(T))
  d = similar(nlp.meta.x0)
  v = fill!(similar(nlp.meta.x0), zero(T))
  s = similar(nlp.meta.x0)
  sc = similar(nlp.meta.x0)
  s_infty = similar(nlp.meta.x0)
  p = similar(nlp.meta.x0)

  return TADAMSolver{T, V, typeof(nlp)}(
    x,
    xt,
    gx,
    m,
    d,
    v,
    s,
    sc,
    s_infty,
    p,
    zero(T),
    false,
    params,
  )
end

function SolverCore.reset!(solver::TADAMSolver{T}) where {T}
  fill!(solver.m, zero(T))
  fill!(solver.v, zero(T))
  solver.step_accepted = false
  return solver
end

SolverCore.reset!(solver::TADAMSolver, ::AbstractNLPModel) = reset!(solver)

@doc (@doc TADAMSolver) function tadam(
  nlp::AbstractNLPModel{T, V};
  η1::Real = get(TADAM_η1, nlp),
  η2::Real = get(TADAM_η2, nlp),
  γ1::Real = get(TADAM_γ1, nlp),
  γ2::Real = get(TADAM_γ2, nlp),
  γ3::Real = get(TADAM_γ3, nlp),
  β1::Real = get(TADAM_β1, nlp),
  β2::Real = get(TADAM_β2, nlp),
  ϵ_v::Real = get(TADAM_ϵ_v, nlp),
  θ1::Real = get(TADAM_θ1, nlp),
  θ2::Real = get(TADAM_θ2, nlp),
  Δmax::Real = get(TADAM_Δmax, nlp),
  kwargs...,
) where {T, V}
  solver = TADAMSolver(
    nlp;
    η1 = convert(T, η1),
    η2 = convert(T, η2),
    γ1 = convert(T, γ1),
    γ2 = convert(T, γ2),
    γ3 = convert(T, γ3),
    β1 = convert(T, β1),
    β2 = convert(T, β2),
    ϵ_v = convert(T, ϵ_v),
    θ1 = convert(T, θ1),
    θ2 = convert(T, θ2),
    Δmax = convert(T, Δmax),
  )
  solver_specific = Dict(:avgβ1max => zero(T))
  stats = GenericExecutionStats(nlp; solver_specific = solver_specific)
  return solve!(solver, nlp, stats; kwargs...)
end

function SolverCore.solve!(
  solver::TADAMSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("tadam should only be called on unconstrained problems.")

  SolverCore.reset!(stats)
  params = solver.params
  η1 = value(params.η1)
  η2 = value(params.η2)
  γ1 = value(params.γ1)
  γ2 = value(params.γ2)
  γ3 = value(params.γ3)
  β1 = value(params.β1)
  β2 = value(params.β2)
  ϵ_v = value(params.ϵ_v)
  θ1 = value(params.θ1)
  θ2 = value(params.θ2)
  Δmax = value(params.Δmax)

  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  xt = solver.xt
  gx = solver.gx
  momentum = solver.m
  d = solver.d
  v = solver.v
  s = solver.s
  p = solver.p

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, gx)
  norm_gx = norm(gx)
  set_dual_residual!(stats, norm_gx)

  solver.Δ = norm_gx / (2^round(log2(norm_gx + one(T))))

  ϵ = atol + rtol * norm_gx
  optimal = norm_gx ≤ ϵ

  if optimal
    @info "Optimal point found at initial point"
    @info log_header(
      [:iter, :f, :dual, :Δ, :β1max],
      [Int, Float64, Float64, Float64, Float64],
      hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :Δ => "Δ", :β1max => "β1max"),
    )
    @info log_row([stats.iter, stats.objective, norm_gx, solver.Δ, zero(T)])
  end

  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :f, :dual, :Δ, :β1max, :status],
      [Int, Float64, Float64, Float64, Float64, String],
      hdr_override = Dict(
        :f => "f(x)",
        :dual => "‖∇f‖",
        :Δ => "Δ",
        :β1max => "β1max",
        :status => "status",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_gx, solver.Δ, zero(T), " "])
  end

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

  @. d = -gx
  @. v = gx^2

  β1max = zero(T)
  avgβ1max = zero(T)
  siter = 1
  oneT = one(T)

  while !done

    # 1. Compute both bounding steps
    solve_tadam_subproblem!(solver.s_infty, d, v, solver.Δ, ϵ_v)
    compute_cauchy!(solver.sc, d, v, solver.Δ, ϵ_v)

    # 2. Find interpolation parameter τ
    tau = find_tau(solver.sc, solver.s_infty, gx, θ1, θ2)

    if tau >= zero(T)
      # Valid step found! Construct the interpolated step.
      @. s = (oneT - tau) * solver.sc + tau * solver.s_infty
      @. xt = x + s

      step_underflow = x == xt
      ΔTk = dot(d, s) - T(0.5) * dot(s .^ 2, sqrt.(v) .+ ϵ_v)

      ft = obj(nlp, xt)
      if ft == -Inf
        set_status!(stats, :unbounded)
        break
      end

      ρk = ΔTk <= eps(T) ? -one(T) : (stats.objective - ft) / ΔTk

      # Standard Trust-Region Update
      if ρk >= η2
        solver.Δ = min(Δmax, γ2 * solver.Δ)
      elseif ρk < η1
        solver.Δ = solver.Δ * γ1
      end

      step_accepted = ρk >= η1
    else
      # 3. Gradient-Related Failure
      # The Dogleg interpolation failed the theoretical conditions. 
      # Shrink momentum strictly to align the model closer to the gradient.
      # DO NOT shrink the trust-region radius here!
      step_underflow = false
      step_accepted = false
      β1max *= γ3
    end

    solver.step_accepted = step_accepted

    # 4. ABSORPTION: Apply EMA Updates ONLY on accepted steps
    if step_accepted
      siter += 1
      x .= xt
      set_objective!(stats, ft)

      # Update buffers using the gx that generated this successful step
      @. momentum = gx * (oneT - β1) + momentum * β1
      @. v = (gx^2 * (oneT - β2) + v * β2 * (oneT - β2^(siter - 1))) / (oneT - β2^siter)
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_gx)
    optimal = norm_gx ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([stats.iter, stats.objective, norm_gx, solver.Δ, β1max, dir_stat])
    end

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

    step_underflow && set_status!(stats, :small_step)
    solver.Δ == zero(T) && set_status!(stats, :exception)

    # 5. CALLBACK: Advances the batch to B_{k+1} if accepted
    callback(nlp, solver, stats)
    done = stats.status != :unknown

    # 6. PREPARATION: Setup geometry for the NEXT iteration
    if step_accepted && !done
      grad!(nlp, x, gx)
      norm_gx = norm(gx)
      set_dual_residual!(stats, norm_gx)

      mdotgx = dot(momentum, gx)
      @. p = momentum - gx
      β1max = find_beta(p, mdotgx, norm_gx, β1, θ1, θ2, siter)
      avgβ1max += β1max

      @. d = -(gx * (oneT - β1max) + momentum * β1max) / (oneT - β1^max(1, siter))

    elseif !step_accepted && !done
      # If rejected, gx and momentum are unchanged. Recompute strictly with shrunk β1max.
      @. d = -(gx * (oneT - β1max) + momentum * β1max) / (oneT - β1^max(1, siter))
    end
  end

  avgβ1max /= max(1, siter - 1)
  stats.solver_specific[:avgβ1max] = avgβ1max
  set_solution!(stats, x)

  return stats
end

"""
    solve_tadam_subproblem!(s, d, v, Δk, ϵ_v)
"""
function solve_tadam_subproblem!(s::V, d::V, v::V, Δk::T, ϵ_v::T) where {V, T}
  # Note: 'd' already contains the negative sign for descent.
  @. s = min(Δk, max(-Δk, d / (sqrt(v) + ϵ_v)))
end

"""
    compute_cauchy!(sc, d, v, Δk, ϵ_v)

Computes the Cauchy point along the gradient-related momentum direction (-d) 
using the exact Subcases 1.1 and 1.2 from the TRAdam paper, utilizing the ∞-norm.
"""
function compute_cauchy!(sc::V, d::V, v::V, Δk::T, ϵ_v::T) where {V, T}
  d_norm_inf = maximum(abs.(d))

  if d_norm_inf <= eps(T)
    fill!(sc, zero(T))
    return
  end

  d_norm_sq = dot(d, d)
  d_H_d = dot(d .^ 2, sqrt.(v) .+ ϵ_v)

  # Case 1: d^T H d > 0 (Always true since H is strictly positive)
  t_star = d_norm_sq / d_H_d

  if t_star * d_norm_inf <= Δk
    # Sub-case 1.1: Inside trust region
    t_C = t_star
  else
    # Sub-case 1.2: Boundary of trust region
    t_C = Δk / d_norm_inf
  end

  # In this Julia code, 'd' is already mathematically negated (d = -gx...)
  # so we move mathematically forward along 'd' by magnitude t_C.
  @. sc = t_C * d
end

"""
    find_tau(sc, s_infty, g, θ1, θ2)

Analytically finds the largest τ ∈ [0,1] satisfying both gradient-related conditions.
"""
function find_tau(sc::V, s_infty::V, g::V, θ1::T, θ2::T) where {V, T}
  w = s_infty .- sc
  τ_min = zero(T)
  τ_max = one(T)

  g_norm_sq = dot(g, g)

  # ---------------------------------------------------------
  # Condition 1: sᵀg ≤ -θ1 ||g||²
  # ---------------------------------------------------------
  A = dot(w, g)
  B = -θ1 * g_norm_sq - dot(sc, g)

  if A > zero(T)
    τ_max = min(τ_max, B / A)
  elseif A < zero(T)
    τ_min = max(τ_min, B / A)
  else
    if B < zero(T)
      return -one(T)
    end
  end

  # ---------------------------------------------------------
  # Condition 2: ||s||² ≤ θ2² ||g||²
  # ---------------------------------------------------------
  a = dot(w, w)
  b = T(2.0) * dot(sc, w)
  c = dot(sc, sc) - (θ2^2 * g_norm_sq)

  if a > eps(T)
    discriminant = b^2 - T(4.0) * a * c
    if discriminant < zero(T)
      return -one(T)
    end

    sq_disc = sqrt(discriminant)
    r1 = (-b - sq_disc) / (T(2.0) * a)
    r2 = (-b + sq_disc) / (T(2.0) * a)

    τ_min = max(τ_min, r1)
    τ_max = min(τ_max, r2)
  elseif c > zero(T)
    return -one(T)
  end

  if τ_min <= τ_max
    return τ_max
  else
    return -one(T)
  end
end

"""
    find_beta(p, mdotgx, norm_gx, β1, θ1, θ2, siter)
"""
function find_beta(p::V, mdotgx::T, norm_gx::T, β1::T, θ1::T, θ2::T, siter::Int) where {T, V}
  n1 = norm_gx^2 - mdotgx
  n2 = norm(p)
  b = (one(T) - β1^siter)

  # Condition 1: Sufficient decrease
  β11 = n1 > zero(T) ? (one(T) - θ1 * b) * norm_gx^2 / n1 : β1
  
  # Condition 2: Upper bound (Fixed triangle inequality algebra)
  β12 = n2 > zero(T) ? (θ2 * b - one(T)) * norm_gx / n2 : β1
  
  return min(β1, min(β11, β12))
end
