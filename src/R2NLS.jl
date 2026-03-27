export R2NLS, R2NLSSolver, R2NLSParameterSet

"""
  R2NLSParameterSet([T=Float64]; η1, η2, θ1, θ2, γ1, γ2, γ3, δ1, σmin, non_mono_size)

Parameter set for the R2NLS solver. Controls algorithmic tolerances and step acceptance.

# Keyword Arguments
- `η1 = eps(T)^(1/4)`: Accept step if actual/predicted reduction ≥ η1 (0 < η1 ≤ η2 < 1).
- `η2 = T(0.95)`: Step is very successful if reduction ≥ η2 (0 < η1 ≤ η2 < 1).
- `θ1 = T(0.5)`: Controls Cauchy step size (0 < θ1 < 1).
- `θ2 = eps(T)^(-1)`: Maximum allowed ratio between the step and the Cauchy step (θ2 > 1).
- `γ1 = T(1.5)`: Regularization increase factor on successful (but not very successful) step (1 < γ1 ≤ γ2).
- `γ2 = T(2.5)`: Regularization increase factor on rejected step (γ1 ≤ γ2).
- `γ3 = T(0.5)`: Regularization increase factor on very successful step (0 < γ3 ≤ 1).
- `δ1 = T(0.5)`: Cauchy point scaling (0 < δ1 < 1). θ1 scales the step size when using the exact Cauchy point, while δ1 scales the step size inexact Cauchy point.
- `σmin = eps(T)`: Smallest allowed regularization.
- `non_mono_size = 1`: Window size for non-monotone acceptance.
- `compute_cauchy_point = false`: Whether to compute the Cauchy point.
- `inexact_cauchy_point = true`: Whether to use an inexact Cauchy point.
"""
struct R2NLSParameterSet{T} <: AbstractParameterSet
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  θ1::Parameter{T, RealInterval{T}}
  θ2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  γ3::Parameter{T, RealInterval{T}}
  δ1::Parameter{T, RealInterval{T}}
  σmin::Parameter{T, RealInterval{T}}
  non_mono_size::Parameter{Int, IntegerRange{Int}}
  compute_cauchy_point::Parameter{Bool, BinaryRange{Bool}}
  inexact_cauchy_point::Parameter{Bool, BinaryRange{Bool}}
end

# Default parameter values
const R2NLS_η1 = DefaultParameter(nls -> begin
  T = eltype(nls.meta.x0)
  T(eps(T))^(T(1)/T(4))
end, "eps(T)^(1/4)")
const R2NLS_η2 = DefaultParameter(nls -> eltype(nls.meta.x0)(0.95), "T(0.95)")
const R2NLS_θ1 = DefaultParameter(nls -> eltype(nls.meta.x0)(0.5), "T(0.5)")
const R2NLS_θ2 = DefaultParameter(nls -> inv(eps(eltype(nls.meta.x0))), "eps(T)^(-1)")
const R2NLS_γ1 = DefaultParameter(nls -> eltype(nls.meta.x0)(1.5), "T(1.5)")
const R2NLS_γ2 = DefaultParameter(nls -> eltype(nls.meta.x0)(2.5), "T(2.5)")
const R2NLS_γ3 = DefaultParameter(nls -> eltype(nls.meta.x0)(0.5), "T(0.5)")
const R2NLS_δ1 = DefaultParameter(nls -> eltype(nls.meta.x0)(0.5), "T(0.5)")
const R2NLS_σmin = DefaultParameter(nls -> eps(eltype(nls.meta.x0)), "eps(T)")
const R2NLS_non_mono_size = DefaultParameter(1)
const R2NLS_compute_cauchy_point = DefaultParameter(false)
const R2NLS_inexact_cauchy_point = DefaultParameter(true)

function R2NLSParameterSet(
  nls::AbstractNLSModel;
  η1::T = get(R2NLS_η1, nls),
  η2::T = get(R2NLS_η2, nls),
  θ1::T = get(R2NLS_θ1, nls),
  θ2::T = get(R2NLS_θ2, nls),
  γ1::T = get(R2NLS_γ1, nls),
  γ2::T = get(R2NLS_γ2, nls),
  γ3::T = get(R2NLS_γ3, nls),
  δ1::T = get(R2NLS_δ1, nls),
  σmin::T = get(R2NLS_σmin, nls),
  non_mono_size::Int = get(R2NLS_non_mono_size, nls),
  compute_cauchy_point::Bool = get(R2NLS_compute_cauchy_point, nls),
  inexact_cauchy_point::Bool = get(R2NLS_inexact_cauchy_point, nls),
) where {T}
  @assert zero(T) < θ1 < one(T) "θ1 must satisfy 0 < θ1 < 1"
  @assert θ2 > one(T) "θ2 must satisfy θ2 > 1"
  @assert zero(T) < η1 <= η2 < one(T) "η1, η2 must satisfy 0 < η1 ≤ η2 < 1"
  @assert one(T) < γ1 <= γ2 "γ1, γ2 must satisfy 1 < γ1 ≤ γ2"
  @assert γ3 > zero(T) && γ3 <= one(T) "γ3 must satisfy 0 < γ3 ≤ 1"
  @assert zero(T) < δ1 < one(T) "δ1 must satisfy 0 < δ1 < 1"
  @assert θ1 <= 2(one(T) - δ1) "θ1 must be ≤ 2(1 - δ1) to ensure sufficient decrease condition is compatible with Cauchy point scaling"

  R2NLSParameterSet{T}(
    Parameter(η1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(η2, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(θ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(θ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ1, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ2, RealInterval(one(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(γ3, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(δ1, RealInterval(zero(T), one(T), lower_open = true, upper_open = true)),
    Parameter(σmin, RealInterval(zero(T), T(Inf), lower_open = true, upper_open = true)),
    Parameter(non_mono_size, IntegerRange(1, typemax(Int))),
    Parameter(compute_cauchy_point, BinaryRange()),
    Parameter(inexact_cauchy_point, BinaryRange()),
  )
end

"""
  R2NLS(nls; kwargs...)

An implementation of the Levenberg-Marquardt method with regularization for nonlinear least-squares problems:

    min ½‖F(x)‖²

where `F: ℝⁿ → ℝᵐ` is a vector-valued function defining the least-squares residuals.

For advanced usage, first create a `R2NLSSolver` to preallocate the necessary memory for the algorithm, and then call `solve!`:

    solver = R2NLSSolver(nls)
    solve!(solver, nls; kwargs...)

# Arguments

- `nls::AbstractNLSModel{T, V}` is the nonlinear least-squares model to solve. See `NLPModels.jl` for additional details.

# Keyword Arguments

- `x::V = nls.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute stopping tolerance.
- `rtol::T = √eps(T)`: relative stopping tolerance; the algorithm stops when ‖J(x)ᵀF(x)‖ ≤ atol + rtol * ‖J(x₀)ᵀF(x₀)‖.
- `Fatol::T = √eps(T)`: absolute tolerance for the residual.
- `Frtol::T = eps(T)`: relative tolerance for the residual; the algorithm stops when ‖F(x)‖ ≤ Fatol + Frtol * ‖F(x₀)‖.
- `params::R2NLSParameterSet = R2NLSParameterSet()`: algorithm parameters, see [`R2NLSParameterSet`](@ref).
- `η1::T = $(R2NLS_η1)`: step acceptance parameter, see [`R2NLSParameterSet`](@ref).
- `η2::T = $(R2NLS_η2)`: step acceptance parameter, see [`R2NLSParameterSet`](@ref).
- `θ1::T = $(R2NLS_θ1)`: Cauchy step parameter, see [`R2NLSParameterSet`](@ref).
- `θ2::T = $(R2NLS_θ2)`: Cauchy step parameter, see [`R2NLSParameterSet`](@ref).
- `γ1::T = $(R2NLS_γ1)`: regularization update parameter, see [`R2NLSParameterSet`](@ref).
- `γ2::T = $(R2NLS_γ2)`: regularization update parameter, see [`R2NLSParameterSet`](@ref).
- `γ3::T = $(R2NLS_γ3)`: regularization update parameter, see [`R2NLSParameterSet`](@ref).
- `δ1::T = $(R2NLS_δ1)`: Cauchy point calculation parameter, see [`R2NLSParameterSet`](@ref).
- `σmin::T = $(R2NLS_σmin)`: minimum step parameter, see [`R2NLSParameterSet`](@ref).
- `non_mono_size::Int = $(R2NLS_non_mono_size)`: the size of the non-monotone history. If > 1, the algorithm will use a non-monotone strategy to accept steps.
- `compute_cauchy_point::Bool = false`: if true, safeguards the step size by reverting to the Cauchy point `scp` if the calculated step `s` is too large relative to the Cauchy step (i.e., if `‖s‖ > θ2 * ‖scp‖`).
- `inexact_cauchy_point::Bool = true`: if true and `compute_cauchy_point` is true, the Cauchy point is calculated using a computationally cheaper inexact formula; otherwise, it is calculated using the operator norm of the Jacobian.
- `subsolver = QRMumpsSubsolver`: the subproblem solver type or instance. Pass a type (e.g., `QRMumpsSubsolver`, `LSMRSubsolver`, `LSQRSubsolver`, `CGLSSubsolver`) to let the solver instantiate it, or pass a pre-allocated instance of `AbstractR2NLSSubsolver`.
- `subsolver_verbose::Int = 0`: if > 0, display subsolver iteration details every `subsolver_verbose` iterations (only applicable for iterative subsolvers).
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum allowed time in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, displays iteration details every `verbose` iterations.

# Output

Returns a `GenericExecutionStats` object containing statistics and information about the optimization process (see `SolverCore.jl`).

- `callback`: function called at each iteration, see [`Callback`](https://jso.dev/JSOSolvers.jl/stable/#Callback) section.

# Examples

```jldoctest
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
model = ADNLSModel(F, [-1.2; 1.0], 2)
solver = R2NLSSolver(model)
stats = solve!(solver, model)
# output
"Execution stats: first-order stationary"
```
"""

mutable struct R2NLSSolver{T, V, Sub <: AbstractR2NLSSubsolver{T}} <: AbstractOptimizationSolver
  x::V         # Current iterate x_k
  xt::V        # Trial iterate x_{k+1}
  gx::V        # Gradient of the objective function: J' * F(x)
  r::V        # Residual vector F(x)
  rt::V        # Residual vector at trial point F(xt)
  temp::V      # Temporary vector for intermediate calculations (e.g. J*v)
  subsolver::Sub # The solver for the linear least-squares subproblem
  obj_vec::V   # History of objective values for non-monotone strategy
  subtol::T    # Current tolerance for the subproblem solver
  s::V         # The calculated step direction
  scp::V       # The Cauchy point step
  σ::T         # Regularization parameter (Levenberg-Marquardt parameter)
  params::R2NLSParameterSet{T} # Algorithmic parameters
end

function R2NLSSolver(
  nls::AbstractNLSModel{T, V};
  subsolver::AbstractR2NLSSubsolver{T} = QRMumpsSubsolver(nls), # Default is an INSTANCE
  η1::T = get(R2NLS_η1, nls),
  η2::T = get(R2NLS_η2, nls),
  θ1::T = get(R2NLS_θ1, nls),
  θ2::T = get(R2NLS_θ2, nls),
  γ1::T = get(R2NLS_γ1, nls),
  γ2::T = get(R2NLS_γ2, nls),
  γ3::T = get(R2NLS_γ3, nls),
  δ1::T = get(R2NLS_δ1, nls),
  σmin::T = get(R2NLS_σmin, nls),
  non_mono_size::Int = get(R2NLS_non_mono_size, nls),
  compute_cauchy_point::Bool = get(R2NLS_compute_cauchy_point, nls),
  inexact_cauchy_point::Bool = get(R2NLS_inexact_cauchy_point, nls),
) where {T, V}
  params = R2NLSParameterSet(
    nls;
    η1 = η1,
    η2 = η2,
    θ1 = θ1,
    θ2 = θ2,
    γ1 = γ1,
    γ2 = γ2,
    γ3 = γ3,
    δ1 = δ1,
    σmin = σmin,
    non_mono_size = non_mono_size,
    compute_cauchy_point = compute_cauchy_point,
    inexact_cauchy_point = inexact_cauchy_point,
  )

  value(params.non_mono_size) >= 1 || error("non_mono_size must be greater than or equal to 1")

  nvar = nls.meta.nvar
  nequ = nls.nls_meta.nequ
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  r = V(undef, nequ)
  rt = V(undef, nequ)
  temp = V(undef, nequ)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  obj_vec = fill(typemin(T), value(params.non_mono_size))

  x .= nls.meta.x0

  # We pass the subsolver instance directly into the struct. No if/else checks needed!
  R2NLSSolver(x, xt, gx, r, rt, temp, subsolver, obj_vec, one(T), s, scp, eps(T)^(1/5), params)
end

function SolverCore.reset!(solver::R2NLSSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.σ = eps(T)^(1 / 5)
  solver.subtol = one(T)
  solver
end

function SolverCore.reset!(solver::R2NLSSolver{T}, nls::AbstractNLSModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.σ = eps(T)^(1 / 5)
  solver.subtol = one(T)
  solver
end

@doc (@doc R2NLSSolver) function R2NLS(
  nls::AbstractNLSModel{T, V};
  η1::Real = get(R2NLS_η1, nls),
  η2::Real = get(R2NLS_η2, nls),
  θ1::Real = get(R2NLS_θ1, nls),
  θ2::Real = get(R2NLS_θ2, nls),
  γ1::Real = get(R2NLS_γ1, nls),
  γ2::Real = get(R2NLS_γ2, nls),
  γ3::Real = get(R2NLS_γ3, nls),
  δ1::Real = get(R2NLS_δ1, nls),
  σmin::Real = get(R2NLS_σmin, nls),
  non_mono_size::Int = get(R2NLS_non_mono_size, nls),
  compute_cauchy_point::Bool = get(R2NLS_compute_cauchy_point, nls),
  inexact_cauchy_point::Bool = get(R2NLS_inexact_cauchy_point, nls),
  subsolver::AbstractR2NLSSubsolver = QRMumpsSubsolver(nls),
  kwargs...,
) where {T, V}
  sub_instance = subsolver isa Type ? subsolver(nls) : subsolver
  solver = R2NLSSolver(
    nls;
    η1 = convert(T, η1),
    η2 = convert(T, η2),
    θ1 = convert(T, θ1),
    θ2 = convert(T, θ2),
    γ1 = convert(T, γ1),
    γ2 = convert(T, γ2),
    γ3 = convert(T, γ3),
    δ1 = convert(T, δ1),
    σmin = convert(T, σmin),
    non_mono_size = non_mono_size,
    compute_cauchy_point = compute_cauchy_point,
    inexact_cauchy_point = inexact_cauchy_point,
    subsolver = sub_instance,
  )
  return solve!(solver, nls; kwargs...)
end

function SolverCore.solve!(
  solver::R2NLSSolver{T, V},
  nls::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nls.meta.x0, # user can reset the initial point here, but it will also be reset in the solver
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::T = √eps(T),
  Frtol::T = eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  scp_flag::Bool = true,
  subsolver_verbose::Int = 0,
) where {T, V}
  unconstrained(nls) || error("R2NLS should only be called on unconstrained problems.")
  if !(nls.meta.minimize)
    error("R2NLS only works for minimization problem")
  end

  SolverCore.reset!(stats)
  params = solver.params
  η1 = value(params.η1)
  η2 = value(params.η2)
  θ1 = value(params.θ1)
  θ2 = value(params.θ2)
  γ1 = value(params.γ1)
  γ2 = value(params.γ2)
  γ3 = value(params.γ3)
  δ1 = value(params.δ1)
  σmin = value(params.σmin)
  non_mono_size = value(params.non_mono_size)

  start_time = time()
  set_time!(stats, 0.0)

  n = nls.nls_meta.nvar
  m = nls.nls_meta.nequ

  x = solver.x .= x
  xt = solver.xt
  r, rt = solver.r, solver.rt
  s = solver.s
  scp = solver.scp
  ∇f = solver.gx

  # Ensure subsolver is up to date with initial x
  initialize!(solver.subsolver, nls, x)

  # Get accessor for Jacobian (abstracted away from solver details)
  Jx = get_jacobian(solver.subsolver)

  # Initial Eval
  residual!(nls, x, r)
  resid_norm = norm(r)
  f = resid_norm^2 / 2
  mul!(∇f, Jx', r)
  norm_∇fk = norm(∇f)

  # Heuristic for initial σ

  solver.σ = 2^round(log2(norm_∇fk + 1)) / norm_∇fk

  # Stopping criterion: 
  unbounded = false
  ρk = zero(T)

  ϵ = atol + rtol * norm_∇fk
  ϵF = Fatol + Frtol * resid_norm

  temp = solver.temp

  stationary = norm_∇fk ≤ ϵ
  small_residual = resid_norm ≤ ϵF

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, norm_∇fk)

  if stationary
    @info "Stationary point found at initial point"
    @info log_header(
      [:iter, :resid_norm, :dual, :σ, :ρ],
      [Int, Float64, Float64, Float64, Float64],
      hdr_override = Dict(:resid_norm => "‖F(x)‖", :dual => "‖∇f‖"),
    )
    @info log_row([stats.iter, resid_norm, norm_∇fk, solver.σ, ρk])
  end

  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :resid_norm, :dual, :σ, :ρ, :sub_iter, :dir, :sub_status],
      [Int, Float64, Float64, Float64, Float64, Int, String, String],
      hdr_override = Dict(
        :resid_norm => "‖F(x)‖",
        :dual => "‖∇f‖",
        :sub_iter => "sub_iter",
        :dir => "dir",
        :sub_status => "status",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, solver.σ, ρk, 0, " ", " "])
  end

  set_status!(
    stats,
    get_status(
      nls,
      elapsed_time = stats.elapsed_time,
      optimal = stationary,
      unbounded = unbounded,
      max_eval = max_eval,
      iter = stats.iter,
      small_residual = small_residual,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  solver.subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * solver.subtol))

  callback(nls, solver, stats)

  # retrieve values again in case the user changed them in the callback

  done = stats.status != :unknown
  compute_cauchy_point = value(params.compute_cauchy_point)
  inexact_cauchy_point = value(params.inexact_cauchy_point)

  while !done

    # 1. Solve Subproblem
    # We pass -r as RHS. Subsolver handles its own temp/workspace for this.
    @. temp = -r

    sub_solved, sub_stats, sub_iter =
      solver.subsolver(s, temp, solver.σ, atol, solver.subtol, verbose = subsolver_verbose)

    # 2. Cauchy Point
    if compute_cauchy_point
      if inexact_cauchy_point
        mul!(temp, Jx, ∇f)
        curvature_gn = dot(temp, temp)
        γ_k = curvature_gn / norm_∇fk^2 + solver.σ
        ν_k = 2 * (1 - δ1) / γ_k
      else
        λmax = get_operator_norm(solver.subsolver)
        ν_k = θ1 / (λmax + solver.σ)
      end

      @. scp = -ν_k * ∇f
      if norm(s) > θ2 * norm(scp)
        s .= scp
      end
    end

    # 3. Acceptance
    xt .= x .+ s
    mul!(temp, Jx, s)
    @. temp += r
    pred_f = norm(temp)^2 / 2
    ΔTk = stats.objective - pred_f

    residual!(nls, xt, rt)
    resid_norm_t = norm(rt)
    ft = resid_norm_t^2 / 2

    if non_mono_size > 1
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      ft_max = maximum(solver.obj_vec)
      ρk = (ft_max - ft) / (ft_max - stats.objective + ΔTk)
    else
      ρk = (stats.objective - ft) / ΔTk
    end

    # 4. Update regularization parameters and determine acceptance of the new candidate
    step_accepted = ρk >= η1

    if step_accepted # Step Accepted
      x .= xt
      r .= rt
      f = ft

      # Update Subsolver Jacobian
      update!(solver.subsolver, nls, x)

      resid_norm = resid_norm_t
      mul!(∇f, Jx', r)
      norm_∇fk = norm(∇f)
      set_objective!(stats, f)

      if ρk >= η2
        solver.σ = max(σmin, γ3 * solver.σ)
      else
        solver.σ = γ1 * solver.σ
      end
    else
      solver.σ = γ2 * solver.σ
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    solver.subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * solver.subtol))

    set_dual_residual!(stats, norm_∇fk)

    callback(nls, solver, stats)

    norm_∇fk = stats.dual_feas

    stationary = norm_∇fk ≤ ϵ
    small_residual = 2 * √f ≤ ϵF

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([stats.iter, resid_norm, norm_∇fk, solver.σ, ρk, sub_iter, dir_stat, sub_stats])
    end

    if stats.status == :user
      done = true
    else
      set_status!(
        stats,
        get_status(
          nls,
          elapsed_time = stats.elapsed_time,
          optimal = stationary,
          unbounded = unbounded,
          small_residual = small_residual,
          max_eval = max_eval,
          iter = stats.iter,
          max_iter = max_iter,
          max_time = max_time,
        ),
      )
    end

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end
