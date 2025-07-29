export R2NLS, R2SolverNLS
export QRMumpsSolver

using QRMumps, LinearAlgebra, SparseArrays

abstract type AbstractQRMumpsSolver end

"""
    QRMumpsSolver

A solver structure for handling the linear least-squares subproblems within R2NLS
using the QRMumps package. This structure pre-allocates all necessary memory
for the sparse matrix representation and the factorization.
"""
mutable struct QRMumpsSolver{T} <: AbstractQRMumpsSolver
  # QRMumps structures
  spmat::qrm_spmat{T}
  spfct::qrm_spfct{T}

  # COO storage for the augmented matrix [J; sqrt(σ) * I]
  irn::Vector{Int}
  jcn::Vector{Int}
  val::Vector{T}

  # Augmented RHS vector
  b_aug::Vector{T}

  # Problem dimensions
  m::Int
  n::Int
  nnzj::Int

  function QRMumpsSolver(nlp::AbstractNLSModel{T}) where {T}
    # Safely initialize QRMumps library
    qrm_init()

    # 1. Get problem dimensions and Jacobian structure
    meta_nls = nls_meta(nlp)
    n = nlp.nls_meta.nvar
    m = nlp.nls_meta.nequ
    nnzj = meta_nls.nnzj

    # 2. Allocate COO arrays for the augmented matrix [J; sqrt(σ)I]
    # Total non-zeros = non-zeros in Jacobian (nnzj) + n diagonal entries for the identity block.
    irn = Vector{Int}(undef, nnzj + n)
    jcn = Vector{Int}(undef, nnzj + n)
    val = Vector{T}(undef, nnzj + n)

    # 3. Fill in the sparsity pattern of the Jacobian J(x)
    jac_structure_residual!(nlp, view(irn, 1:nnzj), view(jcn, 1:nnzj))

    # 4. Fill in the sparsity pattern for the √σ·Iₙ block
    # This block lives in rows m+1 to m+n and columns 1 to n.
    @inbounds for i = 1:n
      irn[nnzj + i] = m + i
      jcn[nnzj + i] = i
    end

    # 5. Initialize QRMumps sparse matrix and factorization structures
    spmat = qrm_spmat_init(m + n, n, irn, jcn, val; sym = false)
    spfct = qrm_spfct_init(spmat)
    qrm_analyse!(spmat, spfct; transp='n')

    # 6. Pre-allocate the augmented right-hand-side vector
    b_aug = Vector{T}(undef, m+n)

    # 7. Create the solver object and set a finalizer for safe cleanup.
    solver = new{T}(spmat, spfct, irn, jcn, val, b_aug, m, n, nnzj)

    return solver
  end
  #TODO cleanup after last iteration
end

const R2NLS_allowed_subsolvers = (:cgls, :crls, :lsqr, :lsmr, :qrmumps)

"""
    R2NLS(nlp; kwargs...)
An inexact second-order quadratic regularization method designed specifically for nonlinear least-squares problems.
The objective is to solve
    min ½‖F(x)‖²
where `F: ℝⁿ → ℝᵐ` is a vector-valued function defining the least-squares residuals.
For advanced usage, first create a `R2SolverNLS` to preallocate the necessary memory for the algorithm, and then call `solve!`:
    solver = R2SolverNLS(nlp)
    solve!(solver, nlp; kwargs...)
# Arguments
- `nlp::AbstractNLSModel{T, V}` is the nonlinear least-squares model to solve. See `NLPModels.jl` for additional details.
# Keyword Arguments
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance; the algorithm stops when ‖J(x)ᵀF(x)‖ ≤ atol + rtol * ‖J(x₀)ᵀF(x₀)‖.
- `η1 =T(0.0001) eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `θ1 = T(0.5)`, `θ2 = eps(T)^-1`: Cauchy step parameters.
- `γ1 = T(1.5)`, `γ2 = T(2.5)`, `γ3 = T(0.5)`: regularization update parameters.
- `δ1 = T(0.5)`: used for Cauchy point calculate.
- `σmin = eps(T)`: minimum step parameter for the R2NLS algorithm.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum allowed time in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, displays iteration details every `verbose` iterations.
- `scp_flag::Bool = true`: if true, we compare the norm of the calculate step with `θ2 * norm(scp)`, each iteration, selecting the smaller step.
- `subsolver::Symbol = :lsmr`: method used as subproblem solver, see `JSOSolvers.R2N_allowed_subsolvers` for a list.
- `subsolver_verbose::Int = 0`: if > 0, displays subsolver iteration details every `subsolver_verbose` iterations when a KrylovWorkspace type is selected.
- `non_mono_size = 1`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.

# Output
Returns a `GenericExecutionStats` object containing statistics and information about the optimization process (see `SolverCore.jl`).
# Callback
$(Callback_docstring)
# Examples
```jldoctest
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
model = ADNLSModel(F, [-1.2; 1.0], 2)
stats = R2NLS(model)
# output
"Execution stats: first-order stationary"
```
```jldoctest
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
model = ADNLSModel(F, [-1.2; 1.0], 2)
solver = R2SolverNLS(model)
stats = solve!(solver, model)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct R2SolverNLS{
  T,
  V,
  Op <: AbstractLinearOperator{T},
  Sub <: Union{KrylovWorkspace{T, T, V}, QRMumpsSolver{T}},
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
  ls_subsolver::Sub
  obj_vec::V # used for non-monotone behaviour
  subtol::T
  s::V
  scp::V
  σ::T
end

function R2SolverNLS(
  nlp::AbstractNLSModel{T, V};
  non_mono_size = 1,
  subsolver::Symbol = :lsmr,
) where {T, V}
  subsolver in R2NLS_allowed_subsolvers ||
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
  # Jx = jac_op_residual!(nlp, x, Jv, Jtv) # todo jacobean when QRMumps is used jac_residual!(nlp, x, Jv, Jtv)
  Op = typeof(Jx)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  σ = eps(T)^(1 / 5)

  if subsolver == :qrmumps #TODO do we need Jv anf Jtv for QRMumps?
    #allocate Jv and Jtv zero length
    # Jv = V(undef, 0)
    # Jtv = V(undef, 0)
    Jx = jac_residual!(nlp, x, Jv, Jtv)
    #how do I Jx
    ls_subsolver = QRMumpsSolver(nlp)
  else
    Jx = jac_op_residual!(nlp, x, Jv, Jtv)
    ls_subsolver = krylov_workspace(Val(subsolver), nequ, nvar, V)
  end
  Sub = typeof(ls_subsolver)

  subtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)

  return R2SolverNLS{T, V, Op, Sub}(
    x,
    xt,
    temp,
    gx,
    Fx,
    rt,
    Jv,
    Jtv,
    Jx,
    ls_subsolver,
    obj_vec,
    subtol,
    s,
    scp,
    σ,
  )
end

function SolverCore.reset!(solver::R2SolverNLS{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end
function SolverCore.reset!(solver::R2SolverNLS{T}, nlp::AbstractNLSModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end

@doc (@doc R2SolverNLS) function R2NLS(
  nlp::AbstractNLSModel{T, V};
  subsolver::Symbol = :lsmr,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2SolverNLS(nlp; non_mono_size = non_mono_size, subsolver = subsolver)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.solve!(
  solver::R2SolverNLS{T, V},
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
  scp_flag::Bool = true,
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
  ls_subsolver = solver.ls_subsolver
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
  cp_step_log = " "
  if verbose > 0 && mod(stats.iter, verbose) == 0
    
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
    @info log_row([stats.iter, stats.objective, norm_∇fk, σk, ρk, 0, " ", " ", " "])
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
  ν_k = one(T)

  while !done

    # Compute the Cauchy step.
    mul!(temp, Jx, ∇f) # temp <- Jx'*∇f
    curv = dot(temp, temp) # curv = ∇f' Jx'Jx *∇f
    slope = σk * norm_∇fk^2 # slope= σ * ||∇f||^2    
    γ_k = (curv + slope) / norm_∇fk^2
    temp .= .-r
    solver.σ = σk

    if γ_k > 0
      ν_k = 2*(1-δ1) / (γ_k)
      cp_step_log = "α_k"
      # Compute the step s.
      subsolver_solved, sub_stats, subiter =
        subsolve!(ls_subsolver, solver, nlp, s, atol, n, m, max_time, subsolver_verbose)
      if scp_flag 
        # Based on the flag, scp is calcualted
        scp .= -ν_k * ∇f
        if norm(s) > θ2 * norm(scp)
          s .= scp
        end
      end
    else  # when zero curvature occures
      # we have to calcualte the scp, since we have encounter a negative curvature
      λmax, found_λ = opnorm(Jx)
      found_λ || error("operator norm computation failed")
      cp_step_log = "ν_k"
      ν_k = θ1 / (λmax + σk)
      scp .= -ν_k * ∇f
      s .= scp
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
      #TODO if QRMumps is used, we have to update Jx else implicitly for other solvers

      # update Jx implicitly
      x .= xt
      r .= rt
      f = fck
      grad!(nlp, x, ∇f, r, recompute = false)
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
    norm_∇fk = stats.dual_feas

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

# Dispatch for KrylovWorkspace
function subsolve!(
  ls_subsolver::KrylovWorkspace,
  R2NLS::R2SolverNLS,
  nlp,
  s,
  atol,
  n,
  m,
  max_time,
  subsolver_verbose,
)
  krylov_solve!(
    ls_subsolver,
    R2NLS.Jx,
    R2NLS.temp,
    atol = atol,
    rtol = R2NLS.subtol,
    λ = √(R2NLS.σ),  # λ ≥ 0 is a regularization parameter.
    itmax = max(2 * (n + m), 50),
    # timemax = max_time - R2SolverNLS.stats.elapsed_time,
    verbose = subsolver_verbose,
  )
  s .= ls_subsolver.x
  return Krylov.issolved(ls_subsolver), ls_subsolver.stats.status, ls_subsolver.stats.niter
end

# Dispatch for QRMumpsSolver
function subsolve!(
  ls::QRMumpsSolver,
  R2NLS::R2SolverNLS,
  nlp,
  s,
  atol,
  n,
  m,
  max_time,
  subsolver_verbose,
)

  # 1. Update Jacobian values at the current point x
  jac_coord_residual!(nlp, R2NLS.x, view(ls.val, 1:ls.nnzj))

  # 2. Update regularization parameter σ
  sqrt_σ = sqrt(R2NLS.σ)
  @inbounds for i = 1:n
    ls.val[ls.nnzj + i] = sqrt_σ
  end

  # 3. Build the augmented right-hand side vector: b_aug = [-F(x); 0]
  ls.b_aug[1:m] .= R2NLS.temp # -F(x)
  fill!(view(ls.b_aug,(m+1):(m+n)), zero(eltype(ls.b_aug))) # we have to do this for some reason #Applying all of its Householder (or Givens) transforms to the entire RHS vector b_aug—i.e. computing QTbQTb.
  # Update spmat
  qrm_update!(ls.spmat, ls.val)

  # 4. Solve the least-squares system
  qrm_factorize!(ls.spmat, ls.spfct; transp='n')
  qrm_apply!(ls.spfct, ls.b_aug; transp='t') 
  qrm_solve!(ls.spfct, ls.b_aug, s; transp='n')
  # qrm_least_squares!(ls.spmat, ls.b_aug, s)
  
  # 5. Return status. For a direct solver, we assume success.
  return true, "QRMumps", 1
end
