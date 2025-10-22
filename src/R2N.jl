export R2N, R2NSolver, R2NParamaterSet
export ShiftedLBFGSSolver, MA97Solver
export ShiftedOperator

using LinearOperators, LinearAlgebra
using SparseArrays
using HSL

#TODO move to LinearOperators
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
  return ShiftedOperator{T, Vector{T}, OpH}(H, zero(T), size(H, 1), H.symmetric, H.hermitian)
end

# Define required properties for AbstractLinearOperator
Base.size(A::ShiftedOperator) = (A.n, A.n)
LinearAlgebra.isreal(A::ShiftedOperator{T}) where {T <: Real} = true
LinearOperators.issymmetric(A::ShiftedOperator) = A.symmetric
LinearOperators.ishermitian(A::ShiftedOperator) = A.hermitian

# Define the core multiplication rules: y = (H + σI)x
function LinearAlgebra.mul!(
  y::AbstractVecOrMat,
  A::ShiftedOperator{T, V, OpH},
  x::AbstractVecOrMat{T},
) where {T, V, OpH}
  mul!(y, A.H, x)
  LinearAlgebra.axpy!(A.σ, x, y)
  return y
end

# function LinearAlgebra.mul!(y::V, A::ShiftedOperator{T, V}, x::V) where {T, V}
#   # y = Hx + σx
#   mul!(y, A.H, x)
#   axpy!(A.σ, x, y) # y += A.σ * x
#   return y
# end

# function LinearAlgebra.mul!(y::V, A::ShiftedOperator{T, V}, x::V, α::Number, β::Number) where {T, V}
#   # y = α(Hx + σx) + βy
#   mul!(y, A.H, x, α, β) # y = α*Hx + β*y
#   axpy!(α * A.σ, x, y)  # y += α*A.σ*x
#   return y
# end

"""
    R2NParameterSet([T=Float64]; θ1, θ2, η1, η2, γ1, γ2, γ3, σmin, non_mono_size)

Parameter set for the R2N solver. Controls algorithmic tolerances and step acceptance.

# Keyword Arguments
- `θ1 = T(0.5)`: Cauchy step parameter (0 < θ1 < 1).
- `θ2 = T(2.0)`: Cauchy step parameter (θ2 > 1).
- `η1 = eps(T)^(1/4)`: Step acceptance parameter (0 < η1 ≤ η2 < 1).
- `η2 = T(0.95)`: Step acceptance parameter (0 < η1 ≤ η2 < 1).
- `γ1 = T(1.5)`: Regularization update parameter (1 < γ1 ≤ γ2).
- `γ2 = T(2.5)`: Regularization update parameter (γ1 ≤ γ2).
- `γ3 = T(0.5)`: Regularization update parameter (0 < γ3 ≤ 1).
- `δ1 = T(0.5)`: Cauchy point calculation parameter.
- `σmin = eps(T)`: Minimum step parameter.
- `non_mono_size = 1`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.
"""
struct R2NParameterSet{T} <: AbstractParameterSet
  θ1::Parameter{T, RealInterval{T}}
  θ2::Parameter{T, RealInterval{T}}
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  γ3::Parameter{T, RealInterval{T}}
  δ1::Parameter{T, RealInterval{T}}
  σmin::Parameter{T, RealInterval{T}}
  non_mono_size::Parameter{Int, IntegerRange{Int}}
end

# Default parameter values
const R2N_θ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_θ2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(2.0), "T(2.0)")
const R2N_η1 = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))^(T(1) / T(4))
end, "eps(T)^(1/4)")
const R2N_η2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.95), "T(0.95)")
const R2N_γ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(1.5), "T(1.5)")
const R2N_γ2 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(2.5), "T(2.5)")
const R2N_γ3 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_δ1 = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.5), "T(0.5)")
const R2N_σmin = DefaultParameter(nlp -> begin
  T = eltype(nlp.meta.x0)
  T(eps(T))
end, "eps(T)")
const R2N_non_mono_size = DefaultParameter(1)

function R2NParameterSet(
  nlp::AbstractNLPModel;
  θ1::T = get(R2N_θ1, nlp),
  θ2::T = get(R2N_θ2, nlp),
  η1::T = get(R2N_η1, nlp),
  η2::T = get(R2N_η2, nlp),
  γ1::T = get(R2N_γ1, nlp),
  γ2::T = get(R2N_γ2, nlp),
  γ3::T = get(R2N_γ3, nlp),
  δ1::T = get(R2N_δ1, nlp),
  σmin::T = get(R2N_σmin, nlp),
  non_mono_size::Int = get(R2N_non_mono_size, nlp),
) where {T}
  R2NParameterSet{T}(
    Parameter(θ1, RealInterval(zero(T), one(T))),
    Parameter(θ2, RealInterval(one(T), T(Inf))),
    Parameter(η1, RealInterval(zero(T), one(T))),
    Parameter(η2, RealInterval(zero(T), one(T))),
    Parameter(γ1, RealInterval(one(T), T(Inf))),
    Parameter(γ2, RealInterval(one(T), T(Inf))),
    Parameter(γ3, RealInterval(zero(T), one(T))),
    Parameter(δ1, RealInterval(zero(T), one(T))),
    Parameter(σmin, RealInterval(zero(T), T(Inf))),
    Parameter(non_mono_size, IntegerRange(1, typemax(Int))),
  )
end

abstract type AbstractShiftedLBFGSSolver end

mutable struct ShiftedLBFGSSolver <: AbstractShiftedLBFGSSolver #TODO Ask what I can do inside here
  # Shifted LBFGS-specific fields
end

abstract type AbstractMA97Solver end

mutable struct MA97Solver{T} <: AbstractMA97Solver
  # HSL MA97 factorization object
  ma97_obj::Ma97{T}

  # Sparse coordinate representation for (B + σI)
  rows::Vector{Int32} # MA97 prefers Int32
  cols::Vector{Int32}
  vals::Vector{T}

  # Keep track of sizes
  n::Int
  nnzh::Int # Non-zeros in the Hessian B

  function MA97Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
    n = nlp.meta.nvar
    nnzh = nlp.meta.nnzh
    total_nnz = nnzh + n

    # 1. Build the coordinate structure of B + σI
    rows = Vector{Int32}(undef, total_nnz)
    cols = Vector{Int32}(undef, total_nnz)
    hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))
    for i = 1:n
      rows[nnzh + i] = i
      cols[nnzh + i] = i
    end

    # 2. Create a temporary sparse matrix ONCE to establish the final sparsity pattern.
    #    The values (ones) are just placeholders.
    K_pattern = sparse(rows, cols, one(T), n, n)

    # 3. Initialize the Ma97 object using the final CSC pattern
    ma97_obj = ma97_csc(
      K_pattern.n,
      Int32.(K_pattern.colptr),
      Int32.(K_pattern.rowval),
      K_pattern.nzval, # Pass initial values
    )

    # 4. Perform the expensive symbolic analysis here, only ONCE.
    # ma97_analyze!(ma97_obj)

    # 5. Allocate a buffer for the values that will be updated in each iteration
    vals = Vector{T}(undef, total_nnz)

    return new{T}(ma97_obj, rows, cols, vals, n, nnzh)
  end
end

const npc_handler_allowed = [:armijo, :sigma, :prev, :cp]

const R2N_allowed_subsolvers = [:cg, :cr, :minres, :minres_qlp, :shifted_lbfgs, :ma97] #TODO Prof. Orban check if I can add more

"""
    R2N(nlp; kwargs...)

An inexact second-order quadratic regularization method for unconstrained optimization (with shifted L-BFGS or shifted Hessian operator).

For advanced usage, first define a `R2NSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2NSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.



# Keyword arguments
- `params::R2NParameterSet = R2NParameterSet(nlp)`: algorithm parameters, see [`R2NParameterSet`](@ref).
- `η1::T = $(R2N_η1)`: step acceptance parameter, see [`R2NParameterSet`](@ref).
- `η2::T = $(R2N_η2)`: step acceptance parameter, see [`R2NParameterSet`](@ref).
- `θ1::T = $(R2N_θ1)`: Cauchy step parameter, see [`R2NParameterSet`](@ref).
- `θ2::T = $(R2N_θ2)`: Cauchy step parameter, see [`R2NParameterSet`](@ref).
- `γ1::T = $(R2N_γ1)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `γ2::T = $(R2N_γ2)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `γ3::T = $(R2N_γ3)`: regularization update parameter, see [`R2NParameterSet`](@ref).
- `δ1::T = $(R2N_δ1)`: Cauchy point calculation parameter, see [`R2NParameterSet`](@ref).
- `σmin::T = $(R2N_σmin)`: minimum step parameter, see [`R2NParameterSet`](@ref).
- `non_mono_size::Int = $(R2N_non_mono_size)`: the size of the non-monotone behaviour. If > 1, the algorithm will use a non-monotone strategy to accept steps.
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of evaluations of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver::Symbol  = :shifted_lbfgs`: the subsolver to solve the shifted system. The `MinresWorkspace` which solves the shifted linear system exactly at each iteration. Using the exact solver is only possible if `nlp` is an `LBFGSModel`. See `JSOSolvers.R2N_allowed_subsolvers` for a list of available subsolvers.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver if KrylovWorkspace type is selected.
- `scp_flag::Bool = true`: if true, we compare the norm of the calculate step with `θ2 * norm(scp)`, each iteration, selecting the smaller step.
- `npc_handler::Symbol = :armijo`: the non_positive_curve handling strategy.
  - `:armijo`: uses the Armijo rule to handle non-positive curvature.
  - `:sigma`: increase the regularization parameter σ.
  - `:prev`: if subsolver return after first iteration, increase the sigma, but if subsolver return after second iteration, set s_k = s_k^(t-1).
  - `:cp`: set s_k to Cauchy point.
See `JSOSolvers.npc_handler_allowed` for a list of available `npc_handler` strategies.

All algorithmic parameters (θ1, θ2, η1, η2, γ1, γ2, γ3, σmin) can be set via the `params` keyword or individually as shown above. If both are provided, individual keywords take precedence.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
$(Callback_docstring)

# Examples #TODO
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
  Op <: Union{AbstractLinearOperator{T}, AbstractMatrix{T}}, #TODO confirm with Prof. Orban
  ShiftedOp <: Union{ShiftedOperator{T, V, Op}, Nothing}, # for cg and cr solvers
  Sub <: Union{KrylovWorkspace{T, T, V}, ShiftedLBFGSSolver, MA97Solver{T}},
} <: AbstractOptimizationSolver
  x::V
  xt::V
  gx::V
  gn::V
  H::Op
  A::ShiftedOp
  Hs::V
  s::V
  scp::V
  obj_vec::V # used for non-monotone
  r2_subsolver::Sub
  subtol::T
  σ::T
  params::R2NParameterSet{T}
end

function R2NSolver(
  nlp::AbstractNLPModel{T, V};
  η1 = get(R2N_η1, nlp),
  η2 = get(R2N_η2, nlp),
  θ1 = get(R2N_θ1, nlp),
  θ2 = get(R2N_θ2, nlp),
  γ1 = get(R2N_γ1, nlp),
  γ2 = get(R2N_γ2, nlp),
  γ3 = get(R2N_γ3, nlp),
  δ1 = get(R2N_δ1, nlp),
  σmin = get(R2N_σmin, nlp),
  non_mono_size = get(R2N_non_mono_size, nlp),
  subsolver::Symbol = :minres,
) where {T, V}
  params = R2NParameterSet(
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
  subsolver in R2N_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2N_allowed_subsolvers)")

  value(params.non_mono_size) >= 1 || error("non_mono_size must be greater than or equal to 1")

  !(subsolver == :shifted_lbfgs) ||
    (nlp isa LBFGSModel) ||
    error("Unsupported subsolver type, ShiftedLBFGSSolver can only be used by LBFGSModel")

  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  A = nothing

  local H, r2_subsolver
  if subsolver == :ma97
    LIBHSL_isfunctional() || error("HSL library is not functional")
    r2_subsolver = MA97Solver(nlp)
    # H is not used as an operator, so we use a placeholder.
    # The matrix is assembled inside subsolve!
    H = spzeros(T, nvar, nvar)

  else  # not using ma971
    # H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, x, Hs) 
    if subsolver == :shifted_lbfgs
      H = nlp.op
      r2_subsolver = ShiftedLBFGSSolver()
    else
      H = hess_op!(nlp, x, Hs)
      r2_subsolver = krylov_workspace(Val(subsolver), nvar, nvar, V)
      if subsolver in (:cg, :cr)
        A = ShiftedOperator(H)
      end
    end
  end

  Op = typeof(H)
  ShiftedOp = typeof(A)
  σ = zero(T)
  s = V(undef, nvar)
  scp = V(undef, nvar)
  subtol = one(T)
  obj_vec = fill(typemin(T), non_mono_size)
  Sub = typeof(r2_subsolver)

  return R2NSolver{T, V, Op, ShiftedOp, Sub}(
    x,
    xt,
    gx,
    gn,
    H,
    A,
    Hs,
    s,
    scp,
    obj_vec,
    r2_subsolver,
    subtol,
    σ,
    params,
  )
end
#TODO Prof. Orban, check if I need to reset H in reset! function
function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  reset!(solver.H)
  if solver.r2_subsolver isa CgWorkspace || solver.r2_subsolver isa CrWorkspace
    solver.A = ShiftedOperator(solver.H)
  end
  solver
end
function SolverCore.reset!(solver::R2NSolver{T}, nlp::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  # @assert (length(solver.gn) == 0) || isa(nlp, QuasiNewtonModel)
  # solver.H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, solver.x, solver.Hs)
  solver
end

@doc (@doc R2NSolver) function R2N(
  nlp::AbstractNLPModel{T, V};
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
  subsolver::Symbol = :minres,
  kwargs...,
) where {T, V}
  solver = R2NSolver(
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
  solver::R2NSolver{T, V},
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
  subsolver_verbose::Int = 0,
  npc_handler::Symbol = :armijo,
  scp_flag::Bool = true,
) where {T, V}
  unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")
  npc_handler in npc_handler_allowed || error("npc_handler must be one of $(npc_handler_allowed)")

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

  @assert(η1 > 0 && η1 < 1)
  @assert(θ1 > 0 && θ1 < 1)
  @assert(θ2 > 1)
  @assert(γ1 >= 1 && γ1 <= γ2 && γ3 <= 1)
  @assert(δ1 > 0 && δ1 < 1)

  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.meta.nvar
  x = solver.x .= x
  xt = solver.xt
  ∇fk = solver.gx # k-1
  ∇fn = solver.gn #current 
  s = solver.s
  scp = solver.scp
  H = solver.H
  A = solver.A
  Hs = solver.Hs
  σk = solver.σ

  subtol = solver.subtol
  subsolver_solved = false

  set_iter!(stats, 0)
  f0 = obj(nlp, x)
  set_objective!(stats, f0)

  grad!(nlp, x, ∇fk)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
  ρk = zero(T)

  # Stopping criterion: 
  fmin = min(-one(T), f0) / eps(T)
  unbounded = f0 < fmin

  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
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
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  # subtol initialization for subsolver 
  subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
  solver.σ = σk
  solver.subtol = subtol
  r2_subsolver = solver.r2_subsolver

  callback(nlp, solver, stats)
  subtol = solver.subtol
  σk = solver.σ

  done = stats.status != :unknown

  ν_k = one(T) # used for scp calculation
  γ_k = zero(T)
  scp_recal = true

  while !done
    scp_recal = true # Recalculate the Cauchy point if needed

    # Compute the Cauchy step.
    # Note that we use buffer values Hs to avoid reallocating memory.
    # mul!(Hs, H, ∇fk) #TODO check for HSL
    if r2_subsolver isa MA97Solver
      hprod!(nlp, x, ∇fk, Hs) # Use exact Hessian-vector product
    else
      mul!(Hs, H, ∇fk) # Use linear operator
    end
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

    # Solving for step direction s_k
    ∇fk .*= -1
    if r2_subsolver isa CgWorkspace || r2_subsolver isa CrWorkspace
      # Update the shift in the operator
      solver.A.σ = σk
      solver.H = H
    end
    subsolver_solved, sub_stats, subiter, npcCount =
      subsolve!(r2_subsolver, solver, nlp, s, zero(T), n, subsolver_verbose)

    if !subsolver_solved
      @warn("Subsolver failed to solve the system")
      # It might be better to increase σ and retry instead of stopping,
      # but for now, we'll stop.
      break
    end
    if !(r2_subsolver isa ShiftedLBFGSSolver) && !(r2_subsolver isa MA97Solver)
      if r2_subsolver.stats.npcCount >= 1  #npc case
        if npc_handler == :armijo #TODO check this logic with Prof. Orban
          c = one(T) * 1e-4 #TODO do I want user to set this value?
          α = one(T) * 1.0
          increase_factor = one(T) * 1.5
          decrease_factor = one(T) * 0.5
          min_alpha = one(T) * 1e-8
          max_alpha = one(T) * 1e2
          dir = r2_subsolver.npc_dir
          f0 = stats.objective
          grad_dir = dot(∇fk, dir)
          # First, try α = 1
          xt .= x .+ α * dir
          fck = obj(nlp, xt)
          if fck > f0 + c * α * grad_dir
            # Backward tracking
            while fck > f0 + c * α * grad_dir && α > min_alpha
              α *= decrease_factor
              xt .= x .+ α * dir
              fck = obj(nlp, xt)
            end
          else
            # Forward tracking
            while true
              α_new = α * increase_factor
              xt .= x .+ α_new * dir
              fck_new = obj(nlp, xt)
              if fck_new > f0 + c * α_new * grad_dir || α_new > max_alpha
                break
              end
              α = α_new
              fck = fck_new
            end
          end
          s .= α * dir
        elseif npc_handler == :prev #Cr and cg will return the last iteration s
          s .= r2_subsolver.x
        elseif npc_handler == :cp
          # Cauchy point
          scp .= ν_k * ∇fk # the ∇fk is already negative
          s .= scp
          scp_recal = false # we don't need to recalculate the scp later
        end
      end
    end

    if scp_flag && scp_recal # if the case was cp, then s = scp already
      # Based on the flag, scp is calcualted
      scp .= ν_k * ∇fk # the ∇fk is already negative
      if norm(s) > θ2 * norm(scp)
        s .= scp
      end
    end
    if npc_handler == :sigma && r2_subsolver.stats.npcCount >= 1  # non-positive curvature case happen and the npc_handler is sigma
      step_accepted = false
    else
      # Correctly compute curvature s' * B * s
      #TODO Prof Orban, check if this is correct
      if solver.r2_subsolver isa MA97Solver
        hprod!(nlp, x, s, Hs) # Use exact Hessian-vector product
      else
        mul!(Hs, H, s) # Use linear operator
      end

      curv = dot(s, Hs)
      slope = dot(s, ∇fk) # = -∇fkᵀ s because we flipped the sign of ∇fk

      ΔTk = slope - curv / 2
      xt .= x .+ s
      fck = obj(nlp, xt)

      if non_mono_size > 1  #non-monotone behaviour
        k = mod(stats.iter, non_mono_size) + 1
        solver.obj_vec[k] = stats.objective
        fck_max = maximum(solver.obj_vec)
        ρk = (fck_max - fck) / (fck_max - stats.objective + ΔTk) #TODO Prof. Orban check if this is correct the denominator   
      else
        # Avoid division by zero/negative. If ΔTk <= 0, the model is bad.
        ρk = (ΔTk > 10 * eps(T)) ? (stats.objective - fck) / ΔTk : -one(T)
      end

      # Update regularization parameters and Acceptance of the new candidate
      step_accepted = ρk >= η1
      if step_accepted
        x .= xt
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
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
    set_dual_residual!(stats, norm_∇fk)

    solver.σ = σk
    solver.subtol = subtol

    callback(nlp, solver, stats)

    norm_∇fk = stats.dual_feas # if the user change it, they just change the stats.norm , they also have to change subtol
    σk = solver.σ
    subtol = solver.subtol
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
  nlp::AbstractNLPModel,
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
    rtol = R2N.subtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver),
  r2_subsolver.stats.status,
  r2_subsolver.stats.niter,
  r2_subsolver.stats.npcCount
end

# Dispatch for MinresWorkspace and MinresQlpWorkspace
function subsolve!(
  r2_subsolver::Union{MinresWorkspace{T, V}, MinresQlpWorkspace{T, V}},
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T, V}
  krylov_solve!(
    r2_subsolver,
    R2N.H,
    R2N.gx,
    λ = R2N.σ,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = R2N.subtol,
    verbose = subsolver_verbose,
    linesearch = true,
  )
  s .= r2_subsolver.x
  return Krylov.issolved(r2_subsolver),
  r2_subsolver.stats.status,
  r2_subsolver.stats.niter,
  r2_subsolver.stats.npcCount
end

# Dispatch for ShiftedLBFGSSolver
function subsolve!(
  r2_subsolver::ShiftedLBFGSSolver,
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
)
  ∇f_neg = R2N.gx
  H = R2N.H
  σ = R2N.σ
  solve_shifted_system!(s, H, ∇f_neg, σ)
  return true, :first_order, 1, 0
end

# Dispatch for MA97Solver
function subsolve!(
  r2_subsolver::MA97Solver{T},
  R2N::R2NSolver,
  nlp::AbstractNLPModel,
  s,
  atol,
  n,
  subsolver_verbose,
) where {T}

  # Unpack for clarity
  g = R2N.gx # Note: R2N main loop has g = -∇f
  σ = R2N.σ
  n = r2_subsolver.n
  nnzh = r2_subsolver.nnzh

  # 1. Update the Hessian part of the values array
  hess_coord!(nlp, R2N.x, view(r2_subsolver.vals, 1:nnzh))

  # 2. Update the shift part of the values array
  # The last 'n' entries correspond to the diagonal for σI
  @inbounds for i = 1:n
    r2_subsolver.vals[nnzh + i] = σ
  end

  # 3. Create the sparse matrix K = B + σI in CSC format.
  # The `sparse` function sums up duplicate entries, which is exactly
  # what we need for the diagonal.
  #TODO with Prof. Orban, check if this is efficient
  # K = sparse(r2_subsolver.rows, r2_subsolver.cols, r2_subsolver.vals, n, n)
  

  # 4. Copy the new numerical values into the MA97 object
  copyto!(r2_subsolver.ma97_obj.nzval, K.nzval)

  # 5. Factorize the matrix
  #TODO Prof Orban, do I need this?# I think we only need to do this once

  ma97_factorize!(r2_subsolver.ma97_obj) 
  if r2_subsolver.ma97_obj.info.flag != 0
    @warn("MA97 factorization failed with flag = $(r2_subsolver.ma97_obj.info.flag)")
    return false, :err, 1, 0 # Indicate failure
  end

  # 6. Solve the system (B + σI)s = g, where g = -∇f
  # s = ma97_solve(r2_subsolver.ma97_obj, g) # Solves in-place
  # 6. Solve the system (B + σI)s = g, where g = -∇f
  s .= g  # Copy the right-hand-side (g) into the solution buffer (s) #TODO confirm with Prof. Orban
  ma97_solve!(r2_subsolver.ma97_obj, s) # Solve in-place, overwriting s

  return true, :first_order, 1, 0
end
