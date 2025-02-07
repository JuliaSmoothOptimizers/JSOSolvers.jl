export R2N, R2NSolver
export ShiftedLBFGSSolver

abstract type AbstractShiftedLBFGSSolver end

struct ShiftedLBFGSSolver <: AbstractShiftedLBFGSSolver
  # Shifted LBFGS-specific fields
end


"""
    R2N(nlp; kwargs...)

An inexact second-order quadratic regularization method for unconstrained optimization (with shifted L-BFGS or shifted Hessian operator).

For advanced usage, first define a `R2NSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2NSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `γ1 = T(1/2)`, `γ2 = 1/γ1`: regularization update parameters.
- `σmin = eps(T)`: step parameter for R2N algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, R2N does not use momentum.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = ShiftedLBFGSSolver`: the subsolver to solve the shifted system. Default is `JSOSolvers.ShiftedLBFGSSolver` which is the exact solver.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver if CG is selected.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current gradient;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2N(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = R2NSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""

mutable struct R2NSolver{
  T,
  V,
  Op <: AbstractLinearOperator{T},
  Op2 <: AbstractLinearOperator{T},
  Sub <: Union{KrylovSolver{T, T, V}, ShiftedLBFGSSolver},
} <: AbstractOptimizationSolver
  x::V
  cx::V
  gx::V
  gn::V
  σ::T
  μ::T
  H::Op
  opI::Op2
  Hs::V
  s::V
  obj_vec::V # used for non-monotone behaviour
  subsolver_type::Sub
  cgtol::T
end

function R2NSolver(
  nlp::AbstractNLPModel{T, V};
  non_mono_size = 1,
  subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = MinresSolver,
) where {T, V}
  nvar = nlp.meta.nvar
  x = V(undef, nvar) # vs similar(nlp.meta.x0)
  cx = V(undef, nvar)
  gx = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, x, Hs)
  opI = opEye(T, nvar)
  Op = typeof(H)
  Op2 = typeof(opI)
  σ = zero(T)
  μ = zero(T)
  s = V(undef, nvar)
  cgtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)
  subsolver =
    isa(subsolver_type, Type{ShiftedLBFGSSolver}) ? subsolver_type() : subsolver_type(nvar, nvar, V)

  Sub = typeof(subsolver)
  return R2NSolver{T, V, Op, Op2, Sub}(
    x,
    cx,
    gx,
    gn,
    σ,
    μ,
    H,
    opI,
    Hs,
    s,
    obj_vec,
    subsolver,
    cgtol,
  )
end

function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  reset!(solver.H)
  solver
end
function SolverCore.reset!(solver::R2NSolver{T}, nlp::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  # @assert (length(solver.gn) == 0) || isa(nlp, QuasiNewtonModel)
  solver.H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, solver.x, solver.Hs)
  
  solver
end

@doc (@doc R2NSolver) function R2N(
  nlp::AbstractNLPModel{T, V};
  subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = MinresSolver,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NSolver(nlp; non_mono_size = non_mono_size, subsolver_type = subsolver_type)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.solve!(
  solver::R2NSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = T(0.0001),
  η2 = T(0.001),
  λ = T(2),
  σmin = zero(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  non_mono_size = 1,
) where {T, V}
  unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")
  if non_mono_size < 1
    error("non_mono_size must be greater than or equal to 1")
  end
  if (solver.subsolver_type isa ShiftedLBFGSSolver && !isa(nlp, LBFGSModel))
    error("Unsupported subsolver type, ShiftedLBFGSSolver is only can be used by LBFGSModel")
  end
  @assert(λ > 1)
  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)
  μmin = σmin
  n = nlp.meta.nvar
  x = solver.x .= x
  ck = solver.cx
  ∇fk = solver.gx # k-1
  ∇fn = solver.gn #current 
  s = solver.s
  H = solver.H
  Hs = solver.Hs
  σk = solver.σ
  μk = solver.μ
  cgtol = solver.cgtol

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, ∇fk)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  μk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  σk = μk * norm_∇fk
  ρk = zero(T)

  # Stopping criterion: 
  fmin = min(-one(T), f0) / eps(T)
  unbounded = f0 < fmin

  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ

  if optimal
    @info("Optimal point found at initial point")
    @info log_header(
      [:iter, :f, :grad_norm, :mu, :sigma, :rho, :dir],
      [Int, Float64, Float64, Float64, Float64, Float64, String],
      hdr_override = Dict(
        :f => "f(x)",
        :grad_norm => "‖∇f‖",
        :mu => "μ",
        :sigma => "σ",
        :rho => "ρ",
        :dir => "DIR",
      ),
    )

    # Define and log the row information with corresponding data values
    @info log_row([stats.iter, stats.objective, norm_∇fk, μk, σk, ρk, ""])
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :f, :grad_norm, :mu, :sigma, :rho, :dir],
      [Int, Float64, Float64, Float64, Float64, Float64, String],
      hdr_override = Dict(
        :f => "f(x)",
        :grad_norm => "‖∇f‖",
        :mu => "μ",
        :sigma => "σ",
        :rho => "ρ",
        :dir => "DIR",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, μk, σk, ρk, ""])
  end

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
  cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))

  while !done
    ∇fk .*= -1
    subsolve!(solver, s, zero(T), n, subsolver_verbose)

    slope = dot(s, ∇fk)
    mul!(Hs, H, s)
    curv = dot(s, Hs)

    ΔTk = slope - curv / 2
    ck .= x .+ s
    fck = obj(nlp, ck)
    ϵ = eps(T)
    
    if ΔTk <= 0
      ΔTk += max(one(T), fck) * 10 * ϵ
      if ΔTk <= 0
        stats.status = :neg_pred
        done = true
        continue
      end
    end
    if non_mono_size > 1  #non-monotone behaviour
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      fck_max = maximum(solver.obj_vec)
      ρk = (fck_max - fck) / (fck_max - stats.objective + ΔTk)
    else
      ρk = (stats.objective - fck) / ΔTk
    end

    # Update regularization parameters and Acceptance of the new candidate
    step_accepted = ρk >= η1 && σk >= η2
    if step_accepted
      μk = max(μmin, μk / λ)
      x .= ck
      grad!(nlp, x, ∇fk)
      if isa(nlp, QuasiNewtonModel)
        ∇fn .-= ∇fk
        ∇fn .*= -1  # = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, ∇fn)
        ∇fn .= ∇fk
      end
      set_objective!(stats, fck)
      unbounded = fck < fmin
      norm_∇fk = norm(∇fk)
    else
      μk = μk * λ
      ∇fk .*= -1
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))
    set_dual_residual!(stats, norm_∇fk)

    solver.σ = σk
    solver.μ = μk
    solver.cgtol = cgtol

    callback(nlp, solver, stats)

    norm_∇fk = stats.dual_feas # if the user change it, they just change the stats.norm 
    μk = solver.μ
    cgtol = solver.cgtol
    σk = μk * norm_∇fk

    optimal = norm_∇fk ≤ ϵ

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info log_row([
        stats.iter,            # Current iteration number
        stats.objective,       # Objective function value
        norm_∇fk,              # Gradient norm
        μk,                    # Mu value
        σk,                    # Sigma value
        ρk,                    # Rho value
        step_accepted ? "↘" : "↗", # Step acceptance status
      ])
    end
    if stats.status == :user
      done = true
    else
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
      done = stats.status != :unknown
    end
  end

  set_solution!(stats, x)
  return stats
end

function subsolve!(R2N::R2NSolver, s, atol, n, subsolver_verbose)
  ∇f = R2N.gx
  cgtol = R2N.cgtol
  H = R2N.H
  subsolver_type = R2N.subsolver_type
  σ = R2N.σ
  opI = R2N.opI

  if subsolver_type isa MinresSolver
    minres!(
      subsolver_type,
      H,
      ∇f, #b 
      λ = σ,
      itmax = max(2 * n, 50),
      atol = atol,
      rtol = cgtol,
      verbose = subsolver_verbose,
    )
    s .= subsolver_type.x
    if norm(subsolver_type.x) < atol
      println("X_norm is:" ,norm(subsolver_type.x)," the status is:"     ,subsolver_type.stats.status)
    end
  elseif subsolver_type isa KrylovSolver
    Krylov.solve!(
      subsolver_type,
      H + opI * σ,
      ∇f,
      atol = atol,
      rtol = cgtol,
      itmax = 2 * n,
      verbose = subsolver_verbose,
    )
    s .= subsolver_type.x
  elseif subsolver_type isa ShiftedLBFGSSolver
    solve_shifted_system!(s, H, ∇f, σ)
  else
    error("Unsupported subsolver type")
  end
end
