export R2NLS, R2NLSSolver, R2NLSParameterSet
export QRMumpsSolver

using LinearAlgebra, SparseArrays
using QRMumps, SparseMatricesCOO

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
- `δ1 = T(0.5)`: Cauchy point scaling (0 < δ1 < 1).
- `σmin = eps(T)`: Smallest allowed regularization.
- `non_mono_size = 1`: Window size for non-monotone acceptance.
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
end

# Default parameter values
const R2NLS_η1 = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))^(T(1)/T(4))
end, "eps(T)^(1/4)")
const R2NLS_η2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.95), "T(0.95)")
const R2NLS_θ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2NLS_θ2 = DefaultParameter(nlp -> inv(eps(eltype(nlp.meta.x0))), "eps(T)^(-1)")
const R2NLS_γ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1.5), "T(1.5)")
const R2NLS_γ2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(2.5), "T(2.5)")
const R2NLS_γ3 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2NLS_δ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2NLS_σmin = DefaultParameter(nlp -> eps(eltype(nlp.meta.x0)), "eps(T)")
const R2NLS_non_mono_size = DefaultParameter(1)

function R2NLSParameterSet(
  nlp::AbstractNLSModel;
  η1::T = get(R2NLS_η1, nlp),
  η2::T = get(R2NLS_η2, nlp),
  θ1::T = get(R2NLS_θ1, nlp),
  θ2::T = get(R2NLS_θ2, nlp),
  γ1::T = get(R2NLS_γ1, nlp),
  γ2::T = get(R2NLS_γ2, nlp),
  γ3::T = get(R2NLS_γ3, nlp),
  δ1::T = get(R2NLS_δ1, nlp),
  σmin::T = get(R2NLS_σmin, nlp),
  non_mono_size::Int = get(R2NLS_non_mono_size, nlp),
) where {T}

  @assert zero(T) < θ1 < one(T) "θ1 must satisfy 0 < θ1 < 1"
  @assert θ2 > one(T) "θ2 must satisfy θ2 > 1"
  @assert zero(T) < η1 <= η2 < one(T) "η1, η2 must satisfy 0 < η1 ≤ η2 < 1"
  @assert one(T) < γ1 <= γ2 "γ1, γ2 must satisfy 1 < γ1 ≤ γ2"
  @assert γ3 > zero(T) && γ3 <= one(T) "γ3 must satisfy 0 < γ3 ≤ 1"
  @assert zero(T) < δ1 < one(T) "δ1 must satisfy 0 < δ1 < 1"
  
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
  )
end

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

  closed::Bool    # Avoid double-destroy

  function QRMumpsSolver(nls::AbstractNLSModel{T}) where {T}
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
    qrm_analyse!(spmat, spfct; transp = 'n')

    # 6. Pre-allocate the augmented right-hand-side vector
    b_aug = Vector{T}(undef, m+n)

    # 7. Create the solver object and set a finalizer for safe cleanup.
    # Initialize 'closed' to false
    solver = new{T}(spmat, spfct, irn, jcn, val, b_aug, m, n, nnzj, false)

    function free_qrm(s::QRMumpsSolver)
      if !s.closed
        qrm_spfct_destroy!(s.spfct)
        qrm_spmat_destroy!(s.spmat)
        s.closed = true
      end
    end

    finalizer(free_qrm, solver) #TODO need more tests
    return solver
  end
end


const R2NLS_allowed_subsolvers = (:cgls, :crls, :lsqr, :lsmr, :qrmumps)

"""

  R2NLS(nlp; kwargs...)

An implementation of the Levenberg-Marquardt method with regularization for nonlinear least-squares problems:

  min ½‖F(x)‖²

where `F: ℝⁿ → ℝᵐ` is a vector-valued function defining the least-squares residuals.

For advanced usage, first create a `R2NLSSolver` to preallocate the necessary memory for the algorithm, and then call `solve!`:

  solver = R2NLSSolver(nlp)
  solve!(solver, nlp; kwargs...)

# Arguments

- `nls::AbstractNLSModel{T, V}` is the nonlinear least-squares model to solve. See `NLPModels.jl` for additional details.

# Keyword Arguments

- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: is the absolute stopping tolerance.
- `rtol::T = √eps(T)`: is the relative stopping tolerance; the algorithm stops when ‖J(x)ᵀF(x)‖ ≤ atol + rtol * ‖J(x₀)ᵀF(x₀)‖.
- `Fatol::T = zero(T)`: absolute tolerance for the residual.
- `Frtol::T = zero(T)`: relative tolerance for the residual; the algorithm stops when ‖F(x)‖ ≤ Fatol + Frtol * ‖F(x₀)‖.
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
- `non_mono_size::Int = $(R2NLS_non_mono_size)`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.
- `scp_flag::Bool = true`: if true, compare the norm of the calculated step with `θ2 * norm(scp)` each iteration, selecting the smaller step.
- `subsolver::Symbol = :lsmr`: method used as subproblem solver, see `JSOSolvers.R2NLS_allowed_subsolvers` for a list.
- `subsolver_verbose::Int = 0`: if > 0, display subsolver iteration details every `subsolver_verbose` iterations when a KrylovWorkspace type is selected.
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
mutable struct R2NLSSolver{
  T,
  V,
  Op <: Union{AbstractLinearOperator{T}, SparseMatrixCOO{T, Int}},
  Sub <: Union{KrylovWorkspace{T, T, V}, QRMumpsSolver{T}},
} <: AbstractOptimizationSolver
  x::V         # Current iterate x_k
  xt::V        # Trial iterate x_{k+1}
  temp::V      # Temporary vector for intermediate calculations (e.g. J*v)
  gx::V        # Gradient of the objective function: J' * F(x)
  Fx::V        # Residual vector F(x)
  rt::V        # Residual vector at trial point F(xt)
  Jv::V        # Storage for Jacobian-vector products (J * v)
  Jtv::V       # Storage for Jacobian-transpose-vector products (J' * v)
  Jx::Op       # The Jacobian operator J(x)
  ls_subsolver::Sub # The solver for the linear least-squares subproblem
  obj_vec::V   # History of objective values for non-monotone strategy
  subtol::T    # Current tolerance for the subproblem solver
  s::V         # The calculated step direction
  scp::V       # The Cauchy point step
  σ::T         # Regularization parameter (Levenberg-Marquardt parameter)
  params::R2NLSParameterSet{T} # Algorithmic parameters
end

function R2NLSSolver(
  nlp::AbstractNLSModel{T, V};
  η1::T = get(R2NLS_η1, nlp),
  η2::T = get(R2NLS_η2, nlp),
  θ1::T = get(R2NLS_θ1, nlp),
  θ2::T = get(R2NLS_θ2, nlp),
  γ1::T = get(R2NLS_γ1, nlp),
  γ2::T = get(R2NLS_γ2, nlp),
  γ3::T = get(R2NLS_γ3, nlp),
  δ1::T = get(R2NLS_δ1, nlp),
  σmin::T = get(R2NLS_σmin, nlp),
  non_mono_size::Int = get(R2NLS_non_mono_size, nlp),
  subsolver::Symbol = :lsmr,
) where {T, V}
  params = R2NLSParameterSet(
    nlp;
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
  )
  subsolver in R2NLS_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2NLS_allowed_subsolvers)")
  value(params.non_mono_size) >= 1 || error("non_mono_size must be greater than or equal to 1")

  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ
  x = V(undef, nvar)
  xt = V(undef, nvar)
  temp = V(undef, nequ)
  gx = V(undef, nvar)
  Fx = V(undef, nequ)
  rt = V(undef, nequ)
  Jv = V(undef, subsolver == :qrmumps ? 0 : nequ)
  Jtv = V(undef, subsolver == :qrmumps ? 0 : nvar)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  σ = eps(T)^(1 / 5)
  if subsolver == :qrmumps
    ls_subsolver = QRMumpsSolver(nlp)
    Jx = SparseMatrixCOO(
      nequ,
      nvar,
      ls_subsolver.irn[1:ls_subsolver.nnzj],
      ls_subsolver.jcn[1:ls_subsolver.nnzj],
      ls_subsolver.val[1:ls_subsolver.nnzj],
    )
  else
    ls_subsolver = krylov_workspace(Val(subsolver), nequ, nvar, V)
    Jx = jac_op_residual!(nlp, x, Jv, Jtv)
  end
  Sub = typeof(ls_subsolver)
  Op = typeof(Jx)

  subtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), value(params.non_mono_size))

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
    ls_subsolver,
    obj_vec,
    subtol,
    s,
    scp,
    σ,
    params,
  )
end

function SolverCore.reset!(solver::R2NLSSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.σ = eps(T)^(1 / 5)
  solver.subtol = one(T)
  solver
end

function SolverCore.reset!(solver::R2NLSSolver{T}, nlp::AbstractNLSModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.σ = eps(T)^(1 / 5)
  solver.subtol = one(T)
  solver
end

@doc (@doc R2NLSSolver) function R2NLS(
  nlp::AbstractNLSModel{T, V};
  η1::Real = get(R2NLS_η1, nlp),
  η2::Real = get(R2NLS_η2, nlp),
  θ1::Real = get(R2NLS_θ1, nlp),
  θ2::Real = get(R2NLS_θ2, nlp),
  γ1::Real = get(R2NLS_γ1, nlp),
  γ2::Real = get(R2NLS_γ2, nlp),
  γ3::Real = get(R2NLS_γ3, nlp),
  δ1::Real = get(R2NLS_δ1, nlp),
  σmin::Real = get(R2NLS_σmin, nlp),
  non_mono_size::Int = get(R2NLS_non_mono_size, nlp),
  subsolver::Symbol = :lsmr,
  kwargs...,
) where {T, V}
  solver = R2NLSSolver(
    nlp;
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
    subsolver = subsolver,
  )
  return solve!(solver, nlp; kwargs...)
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
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  scp_flag::Bool = true,
  subsolver_verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("R2NLS should only be called on unconstrained problems.")
  if !(nlp.meta.minimize)
    error("R2NLS only works for minimization problem")
  end

  reset!(stats)
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
  Jx = solver.Jx

  if Jx isa SparseMatrixCOO
    jac_coord_residual!(nlp, x, view(ls_subsolver.val, 1:ls_subsolver.nnzj))
    Jx.vals .= view(ls_subsolver.val, 1:ls_subsolver.nnzj)
  end

  residual!(nlp, x, r)
  resid_norm = norm(r)
  f = resid_norm^2 / 2

  mul!(∇f, Jx', r)

  norm_∇fk = norm(∇f)
  ρk = zero(T)

  # Stopping criterion: 
  unbounded = false

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  ϵ = atol + rtol * norm_∇fk
  ϵF = Fatol + Frtol * resid_norm

  xt = solver.xt
  temp = solver.temp

  stationary = norm_∇fk ≤ ϵ
  small_residual = 2 * √f ≤ ϵF

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, norm_∇fk)

  if stationary
    @info "Optimal point found at initial point"
    @info log_header(
      [:iter, :resid_norm, :dual, :σ, :ρ],
      [Int, Float64, Float64, Float64, Float64],
      hdr_override = Dict(:resid_norm => "‖F(x)‖", :dual => "‖∇f‖"),
    )
    @info log_row([stats.iter, resid_norm, norm_∇fk, σk, ρk])
  end
  cp_step_log = " "
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info log_header(
      [:iter, :resid_norm, :dual, :σ, :ρ, :sub_iter, :dir, :cp_step_log, :sub_status],
      [Int, Float64, Float64, Float64, Float64, Int, String, String, String],
      hdr_override = Dict(
        :resid_norm => "‖F(x)‖",
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
      optimal = stationary,
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

  # retrieve values again in case the user changed them in the callback
  subtol = solver.subtol
  σk = solver.σ

  done = stats.status != :unknown
  ν_k = one(T) # used for scp calculation

  while !done

    # Compute the Cauchy step.
    mul!(temp, Jx, ∇f) # temp <- Jx ∇f
    curv = dot(temp, temp) # curv = ∇f' Jx' Jx ∇f
    slope = σk * norm_∇fk^2 # slope= σ * ||∇f||^2    
    γ_k = (curv + slope) / norm_∇fk^2
    @. temp = - r
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
      @. scp = -ν_k * ∇f
      s .= scp
    end

    # Compute actual vs. predicted reduction.
    xt .= x .+ s
    mul!(temp, Jx, s)
    slope = dot(r, temp)
    curv = dot(temp, temp)

    residual!(nlp, xt, rt)
    resid_norm_t = norm(rt)
    fck = resid_norm_t^2 / 2

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
      if Jx isa SparseMatrixCOO # we need to update the values of Jx in QRMumpsSolver
        jac_coord_residual!(nlp, x, view(ls_subsolver.val, 1:ls_subsolver.nnzj))
        Jx.vals .= view(ls_subsolver.val, 1:ls_subsolver.nnzj)
      end
      
      # update Jx implicitly for other solvers
      x .= xt
      r .= rt
      f = fck
      resid_norm = resid_norm_t
      mul!(∇f, Jx', r) # ∇f = Jx' * r
      set_objective!(stats, fck)
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

    stationary = norm_∇fk ≤ ϵ
    small_residual = 2 * √f ≤ ϵF

    if verbose > 0 && mod(stats.iter, verbose) == 0
      dir_stat = step_accepted ? "↘" : "↗"
      @info log_row([
        stats.iter,
        resid_norm,
        norm_∇fk,
        σk,
        ρk,
        subiter,
        dir_stat,
        cp_step_log,
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

# Dispatch for KrylovWorkspace
function subsolve!(
  ls_subsolver::KrylovWorkspace,
  R2NLS::R2NLSSolver,
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
    # timemax = max_time - R2NLSSolver.stats.elapsed_time,
    verbose = subsolver_verbose,
  )
  s .= ls_subsolver.x
  return Krylov.issolved(ls_subsolver), ls_subsolver.stats.status, ls_subsolver.stats.niter
end

# Dispatch for QRMumpsSolver
function subsolve!(
  ls::QRMumpsSolver,
  R2NLS::R2NLSSolver,
  nlp,
  s,
  atol,
  n,
  m,
  max_time,
  subsolver_verbose,
)

  # 1. Update Jacobian values at the current point x
  # jac_coord_residual!(nlp, R2NLS.x, view(ls.val, 1:ls.nnzj))

  # 2. Update regularization parameter σ
  sqrt_σ = sqrt(R2NLS.σ)
  @inbounds for i = 1:n
    ls.val[ls.nnzj + i] = sqrt_σ
  end

  # 3. Build the augmented right-hand side vector: b_aug = [-F(x); 0]
  ls.b_aug[1:m] .= R2NLS.temp # -F(x)
  fill!(view(ls.b_aug, (m + 1):(m + n)), zero(eltype(ls.b_aug))) # we have to do this for some reason #Applying all of its Householder (or Givens) transforms to the entire RHS vector b_aug—i.e. computing QTbQTb.
  # Update spmat
  qrm_update!(ls.spmat, ls.val)

  # 4. Solve the least-squares system
  qrm_factorize!(ls.spmat, ls.spfct; transp = 'n')
  qrm_apply!(ls.spfct, ls.b_aug; transp = 't')
  qrm_solve!(ls.spfct, ls.b_aug, s; transp = 'n')

  # 5. Return status. For a direct solver, we assume success.
  return true, "QRMumps", 1
end