using Revise
# using ADNLPModels, JSOSolvers, LinearAlgebra
using NLPModels, ADNLPModels
using HSL_jll  # ] dev C:\Users\Farhad\Documents\Github\openblas_HSL_jll.jl-2023.11.7\HSL_jll.jl-2023.11.7
using HSL
using SparseMatricesCOO
using SparseArrays, LinearAlgebra

# TO test HSL_jll
# TODO A version 2.0 of HSL_jll.jl based on a dummy libHSL has been precompiled with Yggdrasil. Therefore HSL_jll.jl is
# a registered Julia package and can be added as a dependency of any Julia package.
using HSL_jll
function LIBHSL_isfunctional()
  @ccall libhsl.LIBHSL_isfunctional()::Bool
end
bool = LIBHSL_isfunctional()

T = Float64

n = 4
x = ones(T, 4)
nlp = ADNLPModel(
  x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
  x,
  name = "Extended Rosenbrock",
)



  σ = 1.0

n = nlp.meta.nvar
nnzh = nlp.meta.nnzh
total_nnz = nnzh + n  # Max number of coordinates

# 1. Build the coordinate structure of B + σI
rows = Vector{Int}(undef, total_nnz)
cols = Vector{Int}(undef, total_nnz)
vals = zeros(T, total_nnz)  # Explicit zero initialization

# 2. Fill Hessian structure and values
hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))
hess_coord!(nlp, x, view(vals, 1:nnzh))

# 3. Fill diagonal for σI (structure and values)
@inbounds for i = 1:n
    rows[nnzh + i] = i
    cols[nnzh + i] = i
    vals[nnzh + i] = σ
end

# 4. Factorize and solve
ma97_obj = ma97_coord(n, cols, rows, vals)
ma97_factorize!(ma97_obj)

b_aug = randn(T, n) .* 10  # Broadcasting for random vector initialization
x0 = ma97_solve(ma97_obj, b_aug)






sparse(rows, cols, vals)












x = nlp.meta.x0
# 1. get problem dimensions and Hessian structure
meta_nlp = nlp.meta
n = meta_nlp.nvar
nnzh = meta_nlp.nnzh

# 2. Allocate COO arrays for the augmented matrix [H; sqrt(σ)I]
# Total non-zeros = non-zeros in Hessian (nnzh) + n diagonal entries for the identity block.
rows = Vector{Int}(undef, nnzh + n)
cols = Vector{Int}(undef, nnzh + n)
vals = Vector{T}(undef, nnzh + n)

# 3. fill in the structure of the Hessian
hess_structure!(nlp, view(rows, 1:nnzh), view(cols, 1:nnzh))

# 4. Fill in the sparsity pattern for the √σ·Iₙ block
# This block lives in rows n+1 to n+n and columns n+1 to n+n.
@inbounds for i = 1:n
  rows[nnzh + i] = n + i
  cols[nnzh + i] = n + i
end

# 5. Pre-allocate the augmented right-hand-side vector
b_aug = Vector{T}(undef, n)
b_aug[1:n] .= -1.0
#testing the solver

#1. update the hessian values
hess_coord!(nlp, x, view(vals, 1:nnzh))
σ = 1.0
@inbounds for i = 1:n
  vals[nnzh + i] = σ
end


fill!(view(b_aug, (n + 1):(n + n)), zero(eltype(b_aug)))

# Hx = SparseMatrixCOO(
#   nnzh + n,#nvar,
#   nnzh + n, #nvar,
#   rows,
#   cols, #ls_subsolver.jcn[1:ls_subsolver.nnzj],
#   vals, #ls_subsolver.val[1:ls_subsolver.nnzj],
# )
H = sparse(cols, rows, vals)  #TODO add this also to the struct of MA97

LBL = Ma97(H)
ma97_factorize!(LBL)
x_sol = ma97_solve(LBL, b_aug)  # or x = LBL \ rhs

acc = norm(H * x_sol - b_aug) / norm(b_aug)
println("The accuracy of the solution is: $acc")




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#            TEsting to update the values of H
new_vals = ones(nnzh + n) .* 10

H.nzval .= new_vals
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

LBL = Ma97(H)
ma97_factorize!(LBL)
x_sol = ma97_solve(LBL, b_aug)  # or x = LBL \ rhs

acc = norm(H * x_sol - b_aug) / norm(b_aug)
println("The accuracy of the solution is: $acc")









##################OLD way 

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
