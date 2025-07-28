export R2N, R2NSolver
export ShiftedLBFGSSolver
using LinearOperators, LinearAlgebra

# Define a new mutable operator for A = H + σI
mutable struct ShiftedOperator{T, V, OpH <: AbstractLinearOperator{T}} <: AbstractLinearOperator{T}
  H::OpH
  σ::T
  n::Int
  symmetric::Bool
  hermitian::Bool
end

# Constructor for the new operator
function ShiftedOperator(H::OpH) where {T, OpH <: AbstractLinearOperator{T}}
  return ShiftedOperator{T, Vector{T}, OpH}(H, zero(T), H.n, H.symmetric, H.hermitian)
end

# Define required properties for AbstractLinearOperator
Base.size(A::ShiftedOperator) = (A.n, A.n)
LinearAlgebra.isreal(A::ShiftedOperator{T}) where {T <: Real} = true
LinearOperators.is_symmetric(A::ShiftedOperator) = A.symmetric
LinearOperators.is_hermitian(A::ShiftedOperator) = A.hermitian

# Define the core multiplication rules: y = (H + σI)x
function LinearAlgebra.mul!(y::V, A::ShiftedOperator{T, V}, x::V) where {T, V}
  # y = Hx + σx
  mul!(y, A.H, x)
  axpy!(A.σ, x, y) # y += A.σ * x
  return y
end

function LinearAlgebra.mul!(y::V, A::ShiftedOperator{T, V}, x::V, α::Number, β::Number) where {T, V}
  # y = α(Hx + σx) + βy
  mul!(y, A.H, x, α, β) # y = α*Hx + β*y
  axpy!(α * A.σ, x, y)  # y += α*A.σ*x
  return y
end

abstract type AbstractShiftedLBFGSSolver end

struct ShiftedLBFGSSolver <: AbstractShiftedLBFGSSolver
  # Shifted LBFGS-specific fields
end

# const R2N_allowed_subsolvers = [:cg_lanczos_shift, :minres, :shifted_lbfgs]
const R2N_allowed_subsolvers = [:cg, :cr, :minres, :shifted_lbfgs]
#TODO CgLanczosShiftSolver is not implemented yet for negative curvature problems

const npc_handler_allowed = [:armijo, :sigma, :prev, :CP]

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
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver::Symbol  = :shifted_lbfgs`: the subsolver to solve the shifted system. The `MinresSolver` which solves the shifted linear system exactly at each iteration. Using the exact solver is only possible if `nlp` is an `LBFGSModel`.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver if KrylovWorkspace type is selected.
See `JSOSolvers.R2N_allowed_subsolvers` for a list of available `SubSolver`.
- `scp_flag::Bool = true`: if true, we compare the norm of the calculate step with `θ2 * norm(scp)`, each iteration, selecting the smaller step.
- `npc_handler::Symbol = :armijo`: the non_positive_curve handling strategy.
  - `:armijo`: uses the Armijo rule to handle non-positive curvature.
  - `:sigma`: increase the regularization parameter σ.
  - `:prev`: if subsolver return after first iteration, increase the sigma, but if subsolver return after second iteration, set s_k = s_k^(t-1).
  - `:CP`: set s_k to Cauchy point.
See `JSOSolvers.npc_handler_allowed` for a list of available `npc_handler` strategies.
# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.
# Callback
$(Callback_docstring)
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
  OpA <: ShiftedOperator{T, V},
  Sub <: Union{KrylovWorkspace{T, T, V}, ShiftedLBFGSSolver},
} <: AbstractOptimizationSolver
  x::V
  cx::V
  gx::V
  gn::V
  σ::T
  H::Op
  A::OpA # The new combined operator A = H + σI
  Hs::V
  s::V
  scp::V
  obj_vec::V # used for non-monotone
  r2_subsolver::Sub
  cgtol::T
end

function R2NSolver(
  nlp::AbstractNLPModel{T, V};
  non_mono_size = 1,
  subsolver::Symbol = :minres,
) where {T, V}
  subsolver in R2N_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2N_allowed_subsolvers)")

  !(subsolver == :shifted_lbfgs) ||
    (nlp isa LBFGSModel) ||
    error("Unsupported subsolver type, ShiftedLBFGSSolver can only be used by LBFGSModel")

  non_mono_size >= 1 || error("non_mono_size must be greater than or equal to 1")

  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  cx = V(undef, nvar)
  gx = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, x, Hs)
  # Create the single, reusable operator A
  A = ShiftedOperator(H)
  OpA = typeof(A)
  Op = typeof(H)
  σ = zero(T)
  s = V(undef, nvar)
  scp = V(undef, nvar) # Cauchy point
  cgtol = one(T)
  obj_vec = fill(typemin(T), non_mono_size)

  if subsolver == :shifted_lbfgs
    r2_subsolver = ShiftedLBFGSSolver()
  else
    r2_subsolver = krylov_workspace(Val(subsolver), nequ, nvar, V)
  end

  Sub = typeof(r2_subsolver)
  return R2NSolver{T, V, Op, OpA, Sub}(
    x,
    cx,
    gx,
    gn,
    σ,
    H,
    A,
    Hs,
    s,
    scp,
    obj_vec,
    r2_subsolver,
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
  subsolver::Symbol = :minres,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NSolver(nlp; non_mono_size = non_mono_size, subsolver = subsolver)
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
  θ1 = T(0.5),
  θ2 = eps(T)^(-1),
  λ = T(2),
  σmin = zero(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  non_mono_size = 1,
  npc_handler::Symbol = :armijo,
  scp_flag::Bool = true,
) where {T, V}
  unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")
  npc_handler in npc_handler_allowed || error("npc_handler must be one of $(npc_handler_allowed)")
  @assert(λ > 1)

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)
  n = nlp.meta.nvar
  x = solver.x .= x
  ck = solver.cx
  ∇fk = solver.gx # k-1
  ∇fn = solver.gn #current 
  s = solver.s
  scp = solver.scp
  H = solver.H
  Hs = solver.Hs
  A = solver.A
  σk = solver.σ
  cgtol = solver.cgtol
  subsolver_solved = false

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, ∇fk)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  A.σ = σk
  ρk = zero(T)

  # Stopping criterion: 
  fmin = min(-one(T), f0) / eps(T)
  unbounded = f0 < fmin

  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ

  if optimal
    @info("Optimal point found at initial point")
    @info log_header(
      [:iter, :f, :grad_norm, :sigma, :rho, :dir],
      [Int, Float64, Float64, Float64, Float64, String],
      hdr_override = Dict(
        :f => "f(x)",
        :grad_norm => "‖∇f‖",
        :sigma => "σ",
        :rho => "ρ",
        :dir => "DIR",
      ),
    )

    # Define and log the row information with corresponding data values
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk, ""])
  end

  cp_step_log = " "
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :f, :grad_norm, :sigma, :rho, :dir, :cp_step_log, :sub_status],
      [Int, Float64, Float64, Float64, Float64, String, String, String],
      hdr_override = Dict(
        :f => "f(x)",
        :grad_norm => "‖∇f‖",
        :sigma => "σ",
        :rho => "ρ",
        :dir => "DIR",
        :cp_step_log => "Cauchy Step",
        :sub_status => "Subsolver Status",
      ),
    )
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk, "", cp_step_log, ""])
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
  γ_k = zero(T)
  ν_k = one(T)
  scp_recal = true

  while !done
    scp_recal = true # Recalculate the Cauchy point if needed
    # Compute the Cauchy step.
    # Note that we use buffer values Hs to avoid reallocating memory.
    mul!(Hs, H, ∇fk)
    curv = dot(∇fk, Hs)
    slope = σk * norm_∇fk^2 # slope= σ * ||∇f||^2 
    γ_k = (curv + slope) / norm_∇fk^2
    if γ_k < 0
      cp_step_log = "Cauchy step"
      ν_k = 2*(1-δ1) / (γ_k)
    else
      # we have to calcualte the scp, since we have encounter a negative curvature
      λmax, found_λ = opnorm(H)
      cp_step_log = "ν_k"
      ν_k = θ1 / (λmax + σk)
    end

    # Solving for step direction 
    ∇fk .*= -1
    # update the operator A with the current σk
    A.σ = σk
    A.H = H #TODO not really need this ?
    subsolve!(r2_subsolver, solver, s, zero(T), n, subsolver_verbose)
    if !(subsolver == :shifted_lbfgs) #not exact solver
      if r2_subsolver.stats.npcCount >= 1  #npc case
        if npc_handler == :armij
          #TODO
          # elseif npc_handler == :sigma
          #   σk = max(σmin, γ2 * σk)
          #   continue ?
          #   #TODO
        elseif npc_handler == :prev #Cr and cg will return the last iteration s
          s .= r2_subsolver.x
        elseif npc_handler == :CP
          # Cauchy point
          scp .= ν_k * ∇fk # the ∇fk is already negative
          s .= scp
          scp_recal = false # we have already calculated the Cauchy point
        else
          error("Unknown npc_handler: $npc_handler")
        end
      end
    end

    if !subsolver_solved
      @warn("Subsolver failed to solve the shifted system")
      break
    end

    if scp_flag && scp_recal
      # Based on the flag, scp is calcualted
      scp .= ν_k * ∇fk # the ∇fk is already negative
      if norm(s) > θ2 * norm(scp)
        s .= scp
      end
    end

    slope = dot(s, ∇fk) # = -∇fkᵀ s because we flipped the sign of ∇fk
    mul!(Hs, H, s)
    curv = dot(s, Hs)

    ΔTk = slope - curv / 2
    ck .= x .+ s
    fck = obj(nlp, ck)

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
      if ρk >= η2
        σk = max(σmin, γ3 * σk)
      else # η1 ≤ ρk < η2
        σk = min(σmin, γ1 * σk)
      end
    else # η1 > ρk
      σk = max(σmin, γ2 * σk)

      ∇fk .*= -1
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))
    set_dual_residual!(stats, norm_∇fk)

    solver.σ = σk
    solver.cgtol = cgtol
    set_dual_residual!(stats, norm_∇fk)

    callback(nlp, solver, stats)

    norm_∇fk = stats.dual_feas # if the user change it, they just change the stats.norm , they also have to change cgtol
    σk = solver.σ
    cgtol = solver.cgtol
    norm_∇fk = stats.dual_feas

    optimal = norm_∇fk ≤ ϵ
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
# Dispatch for subsolvers KrylovWorkspace: cg and cr
function subsolve!(
  r2_subsolver::KrylovWorkspace{T, T, V},
  R2N::R2NSolver,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T, V}
  krylov_solve!(
    r2_subsolver,
    R2N.A, # Use the ShiftedOperator A
    R2N.gx,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.cgtol,
    verbose = subsolver_verbose,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver), r2_subsolver.stats.status, r2_subsolver.stats.niter
end

# Dispatch for MinresSolver
function subsolve!(r2_subsolver::MinresSolver, R2N::R2NSolver, s, atol, n, subsolver_verbose)
  krylov_solve!(
    r2_subsolver,
    R2N.H,
    R2N.gx,
    λ = R2N.σ,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.cgtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver), r2_subsolver.stats.status, r2_subsolver.stats.niter
end

# Dispatch for KrylovWorkspace
function subsolve!(
  r2_subsolver::CgLanczosShiftSolver,
  R2N::R2NSolver,
  s,
  atol,
  n,
  subsolver_verbose,
)
  krylov_solve!(
    r2_subsolver,
    R2N.H,
    R2N.gx,
    R2N.opI * R2N.σ,  #shift vector σ * I
    λ = R2N.σ,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.cgtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return issolved(r2_subsolver), r2_subsolver.stats.status, r2_subsolver.stats.niter
end

# Dispatch for ShiftedLBFGSSolver
function subsolve!(r2_subsolver::ShiftedLBFGSSolver, R2N::R2NSolver, s, atol, n, subsolver_verbose)
  ∇f_neg = R2N.gx
  H = R2N.H
  σ = R2N.σ
  solve_shifted_system!(s, H, ∇f_neg, σ)
  return true, :first_order, 1
end
