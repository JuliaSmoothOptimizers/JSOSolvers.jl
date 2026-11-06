module JSOSolversHSLExt

using HSL, JSOSolvers, LinearAlgebra, NLPModels
import JSOSolvers:
  AbstractR2NSubsolver,
  MA57R2NSubsolver,
  MA97R2NSubsolver,
  DEFAULT_HSL_DENSE_MAX_NVAR,
  hsl_guard_triggered,
  initialize!,
  update_subsolver!,
  reset_subsolver!,
  finalize_subsolver!,
  is_unsupported,
  get_inertia,
  get_model_hessian,
  get_model_hessian_norm

"""
    HSLR2NSubsolver(nlp; hsl_constructor = ma97_coord,
                    fill_ratio = 0.5, dense_max_nvar = 12_000,
                    max_nvar = typemax(Int), max_nnzh = typemax(Int))

Direct sparse subsolver for [`R2N`](@ref) backed by HSL (`ma57_coord` or `ma97_coord`).

Before touching HSL, a cheap input-size and density guard is applied. If any of

- `nlp.meta.nvar > max_nvar`,
- `nlp.meta.nnzh > max_nnzh` (`nnzh` counts stored lower-triangular Hessian nonzeros), or
- `nnzh > fill_ratio * n(n+1)/2` when `n > dense_max_nvar`

is true, the constructor returns an "unsupported" placeholder: no HSL object is
allocated, no symbolic analysis is run, and [`is_unsupported`](@ref) returns `true`.
`R2N` then short-circuits to status `:exception`. Use this to avoid HSL's eager
symbolic analysis on problems that are too big to factorize (e.g. `n ≈ 120_000`).

# Density guard and problem size

A dense Hessian is cheap enough to factorize at moderate `n`, so the `fill_ratio`
density test only applies once `n > dense_max_nvar`: any problem with
`n ≤ dense_max_nvar` is handed to HSL no matter how dense it is. With the default
`dense_max_nvar = 12_000`, a dense `n = 5_000` problem still runs through HSL.

Set `dense_max_nvar` to the largest `n` for which you accept factorizing a dense
Hessian (work grows like `n³`, storage like `n²`). `dense_max_nvar = 0` restores
the "reject every dense Hessian" behaviour, while `max_nvar`/`max_nnzh` stay hard
caps that apply regardless of density.

Negative `max_nvar`, `max_nnzh` or `dense_max_nvar` raise `ArgumentError`.
`LIBHSL_isfunctional()` is only checked when the guard is not triggered, so
unsupported placeholders can be built without a working HSL library.
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
  dense_max_nvar::Int # largest nvar whose dense Hessian is still accepted
  max_nvar::Int
  max_nnzh::Int
  unsupported::Bool # true when the Hessian is too dense for the direct solver
end

function HSLR2NSubsolver(
  nlp::AbstractNLPModel{T, V};
  hsl_constructor = ma97_coord,
  fill_ratio::Real = 0.5,
  dense_max_nvar::Int = DEFAULT_HSL_DENSE_MAX_NVAR,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) where {T, V}
  max_nvar >= 0 || throw(ArgumentError("max_nvar must be nonnegative"))
  max_nnzh >= 0 || throw(ArgumentError("max_nnzh must be nonnegative"))
  dense_max_nvar >= 0 || throw(ArgumentError("dense_max_nvar must be nonnegative"))
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  fr = T(fill_ratio)

  # Skip building the HSL object (and its eager symbolic analysis) when the
  # problem is too large, or dense *and* larger than `dense_max_nvar`.
  if hsl_guard_triggered(n, nnzh, fr, dense_max_nvar, max_nvar, max_nnzh)
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
      dense_max_nvar,
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
    dense_max_nvar,
    max_nvar,
    max_nnzh,
    false,
  )
  finalizer(finalize_subsolver!, sub)

  return sub
end

# Helper to dispatch safely on the HSL object type
_finalize_hsl_obj!(obj::Ma97) = Base.finalize(obj)
_finalize_hsl_obj!(obj) = nothing # Ma57 holds only Julia-managed memory

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

MA97R2NSubsolver(
  nlp;
  fill_ratio::Real = 0.5,
  dense_max_nvar::Int = DEFAULT_HSL_DENSE_MAX_NVAR,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) = HSLR2NSubsolver(
  nlp;
  hsl_constructor = ma97_coord,
  fill_ratio = fill_ratio,
  dense_max_nvar = dense_max_nvar,
  max_nvar = max_nvar,
  max_nnzh = max_nnzh,
)

MA57R2NSubsolver(
  nlp;
  fill_ratio::Real = 0.5,
  dense_max_nvar::Int = DEFAULT_HSL_DENSE_MAX_NVAR,
  max_nvar::Int = typemax(Int),
  max_nnzh::Int = typemax(Int),
) = HSLR2NSubsolver(
  nlp;
  hsl_constructor = ma57_coord,
  fill_ratio = fill_ratio,
  dense_max_nvar = dense_max_nvar,
  max_nvar = max_nvar,
  max_nnzh = max_nnzh,
)

is_unsupported(sub::HSLR2NSubsolver) = sub.unsupported

function initialize!(sub::HSLR2NSubsolver, nlp, x)
  # Compute the initial Hessian values at x
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return sub
end

function update_subsolver!(sub::HSLR2NSubsolver, nlp, x)
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return sub
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

get_model_hessian(sub::HSLR2NSubsolver) = sub

function get_model_hessian_norm(sub::HSLR2NSubsolver{T}) where {T}
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
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh
  sub.n = n
  sub.nnzh = nnzh

  # 2b. Re-apply the density guard for the new problem. If too dense, skip the
  # expensive rebuild (symbolic analyse) and flag the subsolver as unsupported.
  if hsl_guard_triggered(n, nnzh, sub.fill_ratio, sub.dense_max_nvar, sub.max_nvar, sub.max_nnzh)
    sub.unsupported = true
    return sub
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
  sub.hsl_obj = sub.hsl_constructor(n, sub.cols, sub.rows, sub.vals)
  sub._finalized = false   # ← must reset: object is alive again

  # 6. Resize MA57 workspace if needed
  sub.hsl_constructor === ma57_coord && resize!(sub.work, n)

  # 7. Load actual Hessian values at the starting point
  hess_coord!(nlp, x, view(sub.vals, 1:sub.nnzh))
  return sub
end

end
