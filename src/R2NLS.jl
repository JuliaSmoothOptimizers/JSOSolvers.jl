export R2NLS, R2NLSSolver
export QRMumpsSolver
using QRMumps

abstract type AbstractQRMumpsSolver end

struct QRMumpsSolver <: AbstractQRMumpsSolver
  # Placeholder for QRMumpsSolver
end

const R2NLS_allowed_subsolvers =
  [CglsSolver, CrlsSolver, LsqrSolver, LsmrSolver, MinresSolver, QRMumpsSolver]

"""
    R2NLS(nlp; kwargs...)
An inexact second-order quadratic regularization method designed specifically for nonlinear least-squares problems.
The objective is to solve
    min ½‖F(x)‖²
where `F: ℝⁿ → ℝᵐ` is a vector-valued function defining the least-squares residuals.
For advanced usage, first create a `R2NLSSolver` to preallocate the necessary memory for the algorithm, and then call `solve!`:
    solver = R2NLSSolver(nlp)
    solve!(solver, nlp; kwargs...)
# Arguments
- `nlp::AbstractNLPModel{T, V}` is the nonlinear least-squares model to solve. See `NLPModels.jl` for additional details.
# Keyword Arguments
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance; the algorithm stops when ‖J(x)ᵀF(x)‖ ≤ atol + rtol * ‖J(x₀)ᵀF(x₀)‖.
- `η1 =T(0.0001) eps(T)^(1/4)`, `η2 =T(0.001) T(0.95)`: step acceptance parameters.
- `θ1 = T(0.5)`, `θ2 = T(10)`: Cauchy step parameters.
- `γ1 = T(1.5)`, `γ2 = T(2.5)`, `γ3 = T(0.5)`: regularization update parameters.
- `δ1 = T(0.5)`: used for Cauchy point calculate.
- `σmin = eps(T)`: minimum step parameter for the R2NLS algorithm.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum allowed time in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, displays iteration details every `verbose` iterations.
- `subsolver_type::Union{Type{<:KrylovSolver},Type{QRMumpsSolver}} = LsmrSolver`: the subsolver used to solve the shifted linear system. 
- `subsolver_verbose::Int = 0`: if > 0, displays subsolver iteration details every `subsolver_verbose` iterations when a KrylovSolver type is selected.
- `non_mono_size = 1`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.
See `JSOSolvers.R2N_allowed_subsolvers` for a list of available subsolvers.
# Output
Returns a `GenericExecutionStats` object containing statistics and information about the optimization process (see `SolverCore.jl`).
# Callback
$(Callback_docstring)
# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> [sin(x[1]); cos(x[1])], [1.0])
stats = R2NLS(nlp)
# output
"Execution stats: first-order stationary"
```
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> [sin(x[1]); cos(x[1])], [1.0])
solver = R2NLSSolver(nlp)
stats = solve!(solver, nlp)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct R2NLSSolver{
  T,
  V,
  Op <: AbstractLinearOperator{T},
  Sub <: Union{KrylovSolver{T, T, V}, QRMumpsSolver},
} <: AbstractOptimizationSolver
  x::V
  xt::V
  temp::V
  gx::V
  Fx::V
  rt::V
  Jv::V
  Jtv::V
  Jx::Op
  subsolver::Sub
  obj_vec::V # used for non-monotone behaviour
  subtol::T
  s::V
  scp::V
  σ::T
end

function R2NLSSolver(
  nlp::AbstractNLSModel{T, V};
  non_mono_size = 1,
  subsolver_type::Union{Type{<:KrylovSolver}, Type{QRMumpsSolver}} = LsmrSolver,
) where {T, V}
  subsolver_type in R2NLS_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2NLS_allowed_subsolvers)")
  non_mono_size >= 1 || error("non_mono_size must be greater than or equal to 1")

  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ
  x = V(undef, nvar)
  xt = V(undef, nvar)
  temp = V(undef, nequ)
  gx = V(undef, nvar)
  Fx = V(undef, nequ)
  rt = V(undef, nequ)
  Jv = V(undef, nequ)
  Jtv = V(undef, nvar)
  Jx = jac_op_residual!(nlp, x, Jv, Jtv)
  Op = typeof(Jx)
  if isa(subsolver_type, Type{QRMumpsSolver})
    subsolver =  subsolver_type()
  elseif isa(subsolver_type, Type{MinaresSolver})
    subsolver = subsolver_type(nvar, nvar, V)
  else 
    subsolver = subsolver_type(nvar, nequ, V)
  end
  # subsolver =
  #   isa(subsolver_type, Type{QRMumpsSolver}) ? subsolver_type() : subsolver_type(nequ, nvar, V)
  Sub = typeof(subsolver)

  s = V(undef, nvar)
  scp = V(undef, nvar)
  σ = one(T)

  subtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)

  return R2NLSSolver{T, V, Op, Sub}(
    x,
    xt,
    temp,
    gx,
    Fx,
    rt,
    Jv,
    Jtv,
    Jx,
    subsolver,
    obj_vec,
    subtol,
    s,
    scp,
    σ,
  )
end

function SolverCore.reset!(solver::R2NLSSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end
function SolverCore.reset!(solver::R2NLSSolver{T}, nlp::AbstractNLSModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.Jx = jac_op_residual!(nlp, solver.x, solver.Jv, solver.Jtv)
  # solver.subtol = one(T)
  solver
end

@doc (@doc R2NLSSolver) function R2NLS(
  nlp::AbstractNLSModel{T, V};
  subsolver_type::Union{Type{<:KrylovSolver}, Type{QRMumpsSolver}} = LsmrSolver,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NLSSolver(nlp; non_mono_size = non_mono_size, subsolver_type = subsolver_type)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.solve!(
  solver::R2NLSSolver{T, V},
  nlp::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::T = zero(T),
  Frtol::T = zero(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  θ1 = T(0.5),
  θ2 = eps(T)^(-1),
  γ1 = T(1.5),
  γ2 = T(2.5),
  γ3 = T(0.5),
  δ1 = T(0.5),
  σmin = eps(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  non_mono_size = 1,
) where {T, V}
  unconstrained(nlp) || error("R2NLS should only be called on unconstrained problems.")
  if !(nlp.meta.minimize)
    error("R2NLS only works for minimization problem")
  end

  reset!(stats)
  @assert(η1 > 0 && η1 < 1)
  @assert(θ1 > 0 && θ1 < 1)
  @assert(θ2 > 1)
  @assert(γ1 >= 1 && γ1 <= γ2 && γ3 <= 1)
  @assert(δ1>0 && δ1<1)


  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ
  x = solver.x .= x
  xt = solver.xt
  ∇f = solver.gx # k-1
  subsolver = solver.subsolver
  r, rt = solver.Fx, solver.rt
  s = solver.s
  scp = solver.scp
  subtol = solver.subtol

  σk = solver.σ
  residual!(nlp, x, r)
  f, ∇f = objgrad!(nlp, x, ∇f, r, recompute = false)
  f0 = f

  # preallocate storage for products with Jx and Jx'
  Jx = solver.Jx # jac_op_residual!(nlp, x, Jv, Jtv)
  mul!(∇f, Jx', r)

  norm_∇fk = norm(∇f)
  ρk = zero(T)

  # Stopping criterion: 
  fmin = min(-one(T), f0) / eps(T)
  unbounded = f < fmin

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  ϵ = atol + rtol * norm_∇fk
  ϵF = Fatol + Frtol * 2 * √f
  ν_k = T(0)

  # Preallocate xt.
  xt = solver.xt
  temp = solver.temp

  optimal = norm_∇fk ≤ ϵ
  small_residual = 2 * √f ≤ ϵF

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, norm_∇fk)

  if optimal
    @info "Optimal point found at initial point"
    @info log_header(
      [:iter, :f, :dual, :σ, :ρ],
      [Int, Float64, Float64, Float64, Float64],
      hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖"),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk])
  end

  if verbose > 0 && mod(stats.iter, verbose) == 0
    cp_step_log = " "
    @info log_header(
      [:iter, :f, :dual, :σ, :ρ, :sub_iter, :dir, :cp_step_log, :sub_status],
      [Int, Float64, Float64, Float64, Float64, Int, String, String, String],
      hdr_override = Dict(
        :f => "f(x)",
        :dual => "‖∇f‖",
        :sub_iter => "subiter",
        :dir => "dir",
        :cp_step_log => "cp step",
        :sub_status => "status",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk, 0, " "," ", " "])
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
      small_residual = small_residual,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
  solver.σ = σk
  solver.subtol = subtol

  callback(nlp, solver, stats)

  subtol = solver.subtol
  σk = solver.σ

  done = stats.status != :unknown
  v_k = one(T)
  

  while !done

    # Compute the Cauchy step.
    mul!(temp, Jx, ∇f) # temp <- Jx'*∇f
    curv = dot(temp, temp) # curv = ∇f' Jx'Jx *∇f
    slope = σk * norm_∇fk^2 # slope= σ * ||∇f||^2    
    γ_k = (curv + slope)/ norm_∇fk^2

    #TODO 
    if γ_k > 0
      v_k = 2*(1-δ1)/ (γ_k) 
      if verbose > 0
         cp_step_log = "α_k"
      end
    else
      λmax, found_λ = opnorm(Jx)
      found_λ || error("operator norm computation failed")
      if verbose > 0
         cp_step_log = "ν_k"
      end
      ν_k = θ1 / (λmax + σk)
    end
    
    scp .= -ν_k * ∇f

    temp .= .-r
    # Compute the step s.
    subsolver_solved, sub_stats, subiter =
      subsolve!(subsolver, solver, s, zero(T), n, m, max_time, subsolver_verbose)

    # if (!subsolver_solved) && stats.iter > 0
    #   #TODO
    # end
    if norm(s) > θ2 * norm(scp)
      s .= scp # TODO check if deep copy
    end

    # Compute actual vs. predicted reduction.
    xt .= x .+ s
    mul!(temp, Jx, s)
    slope = dot(r, temp)
    curv = dot(temp, temp)
    residual!(nlp, xt, rt)
    fck = obj(nlp, x, rt, recompute = false)
    
    ΔTk = -slope - curv / 2

    if non_mono_size > 1  #non-monotone behaviour
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      fck_max = maximum(solver.obj_vec)
      ρk = (fck_max - fck) / (fck_max - stats.objective + ΔTk)
    else
      ρk = (stats.objective - fck) / ΔTk
    end
    

    # Update regularization parameters and Acceptance of the new candidate
    step_accepted = ρk >= η1
    if step_accepted
      # update Jx implicitly
      x .= xt
      r .= rt
      f = fck
      grad!(nlp, x, ∇f)
      set_objective!(stats, fck)
      unbounded = fck < fmin
      norm_∇fk = norm(∇f)
      if ρk >= η2
        σk = max(σmin, γ3 * σk)
      else # η1 ≤ ρk < η2
        σk = min(σmin, γ1 * σk)
      end
    else # η1 > ρk
      σk = max(σmin, γ2 * σk)
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))

    solver.σ = σk
    solver.subtol = subtol
    set_dual_residual!(stats, norm_∇fk)

    callback(nlp, solver, stats)

    σk = solver.σ
    subtol = solver.subtol
    norm_∇fk = stats.dual_feas # if the user change it, they just change the stats.norm , they also have to change subtol
    # σk = μk * norm_∇fk

    optimal = norm_∇fk ≤ ϵ
    small_residual = 2 * √f ≤ ϵF

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([
        stats.iter,
        stats.objective,
        norm_∇fk,
        σk,
        ρk,
        subiter,
        cp_step_log,
        dir_stat,
        sub_stats,
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


# Dispatch for MinresSolver
function subsolve!(subsolver::MinresSolver, R2NLS::R2NLSSolver, s, atol, n, m, max_time, subsolver_verbose)
  ∇f_neg = R2NLS.Jx' * R2NLS.temp
  H = R2NLS.Jx' * R2NLS.Jx #TODO allocate 
  σ = √R2NLS.σ
  subtol = R2NLS.subtol

  minres!(
    subsolver,
    H,
    ∇f_neg, # b
    λ = σ/2,
    itmax = max(2 * (n + m), 50),
    atol = atol,
    rtol = subtol,
    verbose = subsolver_verbose,
    # linesearch = true, #TODO update JSOSolvers to use Krylov new workspace
  )
  s .= subsolver.x
  return issolved(subsolver), subsolver.stats.status, subsolver.stats.niter
end

# Dispatch for KrylovSolver
function subsolve!(subsolver::KrylovSolver, R2NLS::R2NLSSolver, s, atol, n, m, max_time,subsolver_verbose)
  Krylov.solve!(
    subsolver,
    R2NLS.Jx,
    R2NLS.temp,
    atol = atol,
    rtol = R2NLS.subtol,
    λ = √(R2NLS.σ) / 2, # or sqrt(σk / 2),  λ ≥ 0 is a regularization parameter.
    itmax = max(2 * (n + m), 50),
    # timemax = max_time - R2NLSSolver.stats.elapsed_time,
    verbose = subsolver_verbose,
  )
  s .= subsolver.x
  return issolved(subsolver), subsolver.stats.status, subsolver.stats.niter
end

# Dispatch for QRMumpsSolver
function subsolve!(subsolver::QRMumpsSolver, R2NLS::R2NLSSolver, s, atol, n, m, max_time,subsolver_verbose)
  #TODO GPU vs CPU 
  QRMumps.qrm_init()
  # Augmented matrix A_aug is (m+n)×n
  A_aug = [
    R2NLS.Jx
    sqrt(R2NLS.σ) * I(n)
  ]      # I(n): identity of size n
  # Augmented right-hand side
  b_aug = [
    -R2NLS.Fx
    zeros(n)
  ]
  spmat = qrm_spmat_init(Matrix(A_aug))    # wrap in QRMumps format
  s = qrm_min_norm(spmat, b_aug)   # min-norm solution of A_aug * s = b_aug
  return true, "QRMumpsSolver", 0 #TODO fix this 
end