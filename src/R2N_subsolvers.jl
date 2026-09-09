using HSL
export ShiftedLBFGSSolver, HSLR2NSubsolver, KrylovR2NSubsolver
export CGR2NSubsolver, CRR2NSubsolver, MinresR2NSubsolver, MinresQlpR2NSubsolver
export AbstractR2NSubsolver
export MA97R2NSubsolver, MA57R2NSubsolver

# ==============================================================================
#   Krylov Subsolver (CG, CR, MINRES)
# ==============================================================================

mutable struct KrylovR2NSubsolver{T, V, Op, W, ShiftOp} <: AbstractR2NSubsolver{T}
  workspace::W
  H::Op           # The Hessian Operator
  A::ShiftOp      # The Shifted Operator (only for CG/CR)
  solver_name::Symbol
  npc_dir::V      # Store NPC direction if needed

  function KrylovR2NSubsolver(nlp::AbstractNLPModel{T, V}, solver_name::Symbol = :cg) where {T, V}
    x_init = nlp.meta.x0
    n = nlp.meta.nvar
    H = isa(nlp, LBFGSModel) ? nlp.op : hess_op(nlp, x_init)

    A = nothing
    A = ShiftedOperator(H)

    workspace = krylov_workspace(Val(solver_name), n, n, V)

    new{T, V, typeof(H), typeof(workspace), typeof(A)}(workspace, H, A, solver_name, V(undef, n))
  end
end

CGR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cg)
CRR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :cr)
MinresR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres) 
MinresQlpR2NSubsolver(nlp) = KrylovR2NSubsolver(nlp, :minres_qlp)

function initialize!(sub::KrylovR2NSubsolver, nlp, x)
  # x here is the live solver.x from the main loop!
  sub.H = isa(nlp, LBFGSModel) ? nlp.op : hess_op(nlp, x)
  sub.A = ShiftedOperator(sub.H)
  return nothing
end

function update_subsolver!(sub::KrylovR2NSubsolver, nlp, x)
  # Standard hess_op updates internally if it holds the NLP reference
  return nothing
end

function (sub::KrylovR2NSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  sub.workspace.stats.niter = 0

  sub.A.data.σ = σ
  krylov_solve!(
    sub.workspace,
    sub.A,
    rhs,
    itmax = max(2 * n, 50),
    atol = atol,
    rtol = rtol,
    verbose = verbose,
    linesearch = true,
  )

  s .= sub.workspace.x
  if isdefined(sub.workspace, :npc_dir)
    sub.npc_dir .= sub.workspace.npc_dir
  end

  # Return the tuple expected by the main loop
  return Krylov.issolved(sub.workspace),
  sub.workspace.stats.status,
  sub.workspace.stats.niter,
  sub.workspace.stats.npcCount
end

get_operator(sub::KrylovR2NSubsolver) = sub.H
has_npc_direction(sub::KrylovR2NSubsolver) =
  isdefined(sub.workspace, :npc_dir) && sub.workspace.stats.npcCount > 0

function get_npc_direction(sub::KrylovR2NSubsolver)
  has_npc_direction(sub) || error("No NPC direction found.")
  return sub.npc_dir
end
function get_operator_norm(sub::KrylovR2NSubsolver)
  # Estimate norm of H. 
  val, _ = LinearOperators.estimate_opnorm(sub.H)
  return val
end

# ==============================================================================
#   Shifted LBFGS Subsolver
# ==============================================================================

mutable struct ShiftedLBFGSSolver{T, Op} <: AbstractR2NSubsolver{T}
  H::Op # The LBFGS Operator

  function ShiftedLBFGSSolver(nlp::AbstractNLPModel{T, V}) where {T, V}
    if !(nlp isa LBFGSModel)
      error("ShiftedLBFGSSolver can only be used by LBFGSModel")
    end
    new{T, typeof(nlp.op)}(nlp.op)
  end
end

# ShiftedLBFGSSolver(nlp) = ShiftedLBFGSSolver(nlp)

initialize!(sub::ShiftedLBFGSSolver, nlp, x) = nothing
update_subsolver!(sub::ShiftedLBFGSSolver, nlp, x) = nothing # LBFGS updates via push! in outer loop

function (sub::ShiftedLBFGSSolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  # rhs is usually -∇f. solve_shifted_system! expects negative gradient
  solve_shifted_system!(s, sub.H, rhs, σ)
  return true, :first_order, 1, 0
end

get_operator(sub::ShiftedLBFGSSolver) = sub.H

function get_operator_norm(sub::ShiftedLBFGSSolver)
  # Estimate norm of H. 
  val, _ = LinearOperators.estimate_opnorm(sub.H)
  return val
end

# ==============================================================================
#   HSL Subsolver (MA97 / MA57)
# ==============================================================================

"""
    HSLR2NSubsolver(nlp; hsl_constructor = ma97_coord,
                    fill_ratio = 0.5, min_matrix_size = 1_000,
                    max_nvar = typemax(Int), max_nnzh = typemax(Int))

Direct sparse subsolver for [`R2N`](@ref) backed by HSL (`ma57_coord` or `ma97_coord`).

Before touching HSL, a cheap input-size and density guard is applied. If any of

- `nlp.meta.nvar > max_nvar`,
- `nlp.meta.nnzh > max_nnzh` (`nnzh` counts stored lower-triangular Hessian nonzeros), or
- `nnzh > fill_ratio * n(n+1)/2` when `n(n+1)/2 > min_matrix_size`

is true, the constructor returns an "unsupported" placeholder: no HSL object is
allocated, no symbolic analysis is run, and [`is_unsupported`](@ref) returns `true`.
`R2N` then short-circuits to status `:exception`. Use this to avoid HSL's eager
symbolic analysis on large or dense problems (e.g. `n ≈ 120_000`).

Negative `max_nvar`/`max_nnzh` raise `ArgumentError`. `LIBHSL_isfunctional()` is only
checked when the guard is not triggered, so unsupported placeholders can be built
without a working HSL library.
"""
mutable struct HSLR2NSubsolver{T, S} <: AbstractR2NSubsolver{T}
  hsl_obj::S
  hsl_constructor::Function
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector{T}
  n::Int
  nnzh::Int
  work::Vector{T} # workspace for solves (used for MA57)
  _finalized::Bool
  fill_ratio::T   # density threshold used by the guard
  min_matrix_size::Int # matrix entry threshold used by the guard
  max_nvar::Int
  max_nnzh::Int
  unsupported::Bool # true when the Hessian is too dense for the direct solver
end

function HSLR2NSubsolver(
  nlp::AbstractNLPModel{T, V};
  hsl_constructor = ma97_coord,
  fill_ratio::Real = 0.5,
  min_matrix_size::Int = 1_000,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) where {T, V}
  max_nvar >= 0 || throw(ArgumentError("max_nvar must be nonnegative"))
  max_nnzh >= 0 || throw(ArgumentError("max_nnzh must be nonnegative"))
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  fr = T(fill_ratio)

  # Cheap O(1) size and density guard. A dense symmetric Hessian has n(n+1)/2
  # nonzeros. Large, dense patterns can make the eager HSL symbolic analysis
  # prohibitively expensive, so skip building the HSL object in that case.
  dense_nnz = (n * (n + 1)) ÷ 2
  if n > max_nvar || nnzh > max_nnzh ||
     (dense_nnz > min_matrix_size && nnzh > fr * dense_nnz)
    return HSLR2NSubsolver{T, Nothing}(
      nothing,
      hsl_constructor,
      Int[],
      Int[],
      T[],
      n,
      nnzh,
      T[],
      true,
      fr,
      min_matrix_size,
      max_nvar,
      max_nnzh,
      true,
    )
  end

  LIBHSL_isfunctional() || error("HSL library is not functional")
  total_nnz = nnzh + n

  rows = Vector{Int}(undef, total_nnz)
  cols = Vector{Int}(undef, total_nnz)
  vals = Vector{T}(undef, total_nnz)

  # Structure analysis must happen in constructor to define the object type S
  hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))

  # Initialize values to zero. Actual computation happens in initialize!
  fill!(vals, zero(T))

  @inbounds for i = 1:n
    rows[nnzh + i] = i
    cols[nnzh + i] = i
    # Diagonal shift will be updated during solve using σ
    vals[nnzh + i] = one(T)
  end

  hsl_obj = hsl_constructor(n, cols, rows, vals)

  if hsl_constructor == ma57_coord
    work = Vector{T}(undef, n * size(nlp.meta.x0, 2))
  else
    work = Vector{T}(undef, 0)
  end

  sub = HSLR2NSubsolver{T, typeof(hsl_obj)}(
    hsl_obj,
    hsl_constructor,
    rows,
    cols,
    vals,
    n,
    nnzh,
    work,
    false,
    fr,
    min_matrix_size,
    max_nvar,
    max_nnzh,
    false,
  )
  finalizer(finalize_subsolver!, sub)
  
  return sub
end

# Helper to dispatch safely on the HSL object type
_finalize_hsl_obj!(obj::Ma97) = Base.finalize(obj)
_finalize_hsl_obj!(obj)       = nothing # Fallback for Ma57 or other types #TODO if MA57 has it add it 


# _finalize_hsl_obj!(obj::Ma57) = Base.finalize(obj)  # was `nothing` → silent leak

 
# Krylov/LBFGS: no-op, so try/finally in R2N is safe for all solvers
function finalize_subsolver!(sub::AbstractR2NSubsolver)
  return nothing
end

"""
    finalize_subsolver!(sub::HSLR2NSubsolver)

Safely triggers the underlying HSL finalizer exactly once.
"""
function finalize_subsolver!(sub::HSLR2NSubsolver)
  sub._finalized && return nothing
  sub._finalized = true
  _finalize_hsl_obj!(sub.hsl_obj)
  return nothing
end

"""
    MA97R2NSubsolver(nlp; fill_ratio = 0.5, min_matrix_size = 1_000,
                    max_nvar = typemax(Int), max_nnzh = typemax(Int))

[`HSLR2NSubsolver`](@ref) using `ma97_coord`. Same guards and keyword semantics.
"""
MA97R2NSubsolver(
  nlp;
  fill_ratio::Real = 0.5,
  min_matrix_size::Int = 1_000,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) =
  HSLR2NSubsolver(
    nlp;
    hsl_constructor = ma97_coord,
    fill_ratio = fill_ratio,
    min_matrix_size = min_matrix_size,
    max_nvar = max_nvar,
    max_nnzh = max_nnzh,
  )

"""
    MA57R2NSubsolver(nlp; fill_ratio = 0.5, min_matrix_size = 1_000,
                    max_nvar = typemax(Int), max_nnzh = typemax(Int))

[`HSLR2NSubsolver`](@ref) using `ma57_coord`. Same guards and keyword semantics.
"""
MA57R2NSubsolver(
  nlp;
  fill_ratio::Real = 0.5,
  min_matrix_size::Int = 1_000,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) =
  HSLR2NSubsolver(
    nlp;
    hsl_constructor = ma57_coord,
    fill_ratio = fill_ratio,
    min_matrix_size = min_matrix_size,
    max_nvar = max_nvar,
    max_nnzh = max_nnzh,
  )

is_unsupported(sub::HSLR2NSubsolver) = sub.unsupported

function initialize!(sub::HSLR2NSubsolver, nlp, x)
  # Compute the initial Hessian values at x
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return nothing
end

function update_subsolver!(sub::HSLR2NSubsolver, nlp, x)
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
end

function get_inertia(sub::HSLR2NSubsolver{T, S}) where {T, S <: Ma97{T}}
  n = sub.n
  num_neg = sub.hsl_obj.info.num_neg
  num_zero = n - sub.hsl_obj.info.matrix_rank
  return num_neg, num_zero
end

function get_inertia(sub::HSLR2NSubsolver{T, S}) where {T, S <: Ma57{T}}
  n = sub.n
  num_neg = sub.hsl_obj.info.num_negative_eigs
  num_zero = n - sub.hsl_obj.info.rank
  return num_neg, num_zero
end

function _hsl_factor_and_solve!(sub::HSLR2NSubsolver{T, S}, g, s) where {T, S <: Ma97{T}}
  ma97_factorize!(sub.hsl_obj)
  if sub.hsl_obj.info.flag < 0
    return false, :subsolver_failed_ma97, 0, 0
  end
  s .= g
  ma97_solve!(sub.hsl_obj, s)
  return true, :first_order, 1, 0
end

function _hsl_factor_and_solve!(sub::HSLR2NSubsolver{T, S}, g, s) where {T, S <: Ma57{T}}
  ma57_factorize!(sub.hsl_obj)
  s .= g
  ma57_solve!(sub.hsl_obj, s, sub.work)
  return true, :first_order, 1, 0
end

function (sub::HSLR2NSubsolver)(s, rhs, σ, atol, rtol, n; verbose = 0)
  # Update diagonal shift in the vals array
  @inbounds for i = 1:n
    sub.vals[sub.nnzh + i] = σ
  end
  return _hsl_factor_and_solve!(sub, rhs, s)
end

get_operator(sub::HSLR2NSubsolver) = sub

function get_operator_norm(sub::HSLR2NSubsolver)
  # Calculate the matrix infinity norm (maximum absolute row sum)
  # Exclude the shift values which are at indices nnzh+1:end
  row_sums = zeros(T, sub.n)
  
  @inbounds for k = 1:sub.nnzh
    i = sub.rows[k]
    j = sub.cols[k]
    v = abs(sub.vals[k])
    
    row_sums[i] += v
    if i != j
      row_sums[j] += v
    end
  end
  
  return maximum(row_sums)
end

# Helper to support `mul!` for HSL subsolver
function LinearAlgebra.mul!(y::AbstractVector, sub::HSLR2NSubsolver, x::AbstractVector)
  coo_sym_prod!(
    view(sub.rows, 1:sub.nnzh),
    view(sub.cols, 1:sub.nnzh),
    view(sub.vals, 1:sub.nnzh),
    x,
    y,
  )
end

function reset_subsolver!(sub::HSLR2NSubsolver{T}, nlp::AbstractNLPModel, x) where {T}
  # 1. Free current C/Fortran object before rebuilding
  finalize_subsolver!(sub)

  # 2. Update dimensions (new problem may differ in size/nnzh)
  n    = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  sub.n    = n
  sub.nnzh = nnzh

  # 2b. Re-apply the density guard for the new problem. If too dense, skip the
  # expensive rebuild (symbolic analyse) and flag the subsolver as unsupported.
  dense_nnz = (n * (n + 1)) ÷ 2
  if n > sub.max_nvar || nnzh > sub.max_nnzh ||
     (dense_nnz > sub.min_matrix_size && nnzh > sub.fill_ratio * dense_nnz)
    sub.unsupported = true
    return nothing
  end
  sub.unsupported = false

  # 3. Resize storage arrays if the new problem is larger or smaller
  total_nnz = nnzh + n
  resize!(sub.rows, total_nnz)
  resize!(sub.cols, total_nnz)
  resize!(sub.vals, total_nnz)

  # 4. Populate sparsity structure for the new problem
  hess_structure!(nlp, view(sub.rows, 1:nnzh), view(sub.cols, 1:nnzh))
  fill!(view(sub.vals, 1:total_nnz), zero(T))
  @inbounds for i = 1:n
    sub.rows[nnzh + i] = i
    sub.cols[nnzh + i] = i
    sub.vals[nnzh + i] = one(T)
  end

  # 5. Recreate the HSL object — triggers symbolic analysis for the new structure
  sub.hsl_obj    = sub.hsl_constructor(n, sub.cols, sub.rows, sub.vals)
  sub._finalized = false   # ← must reset: object is alive again

  # 6. Resize MA57 workspace if needed
  sub.hsl_constructor === ma57_coord && resize!(sub.work, n)

  # 7. Load actual Hessian values at the starting point
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return nothing
end
