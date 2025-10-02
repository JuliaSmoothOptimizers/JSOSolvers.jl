using Revise
# using ADNLPModels, JSOSolvers, LinearAlgebra
using NLPModels, ADNLPModels
using HSL
using SparseMatricesCOO
using SparseArrays, LinearAlgebra


T = Float64

n = 4
x = ones(T, 4)
nlp = ADNLPModel(
    x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
    x,
    name = "Extended Rosenbrock",
)

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
hess_structure!(nlp,view(rows, 1:nnzh), view(cols, 1:nnzh))

# 4. Fill in the sparsity pattern for the √σ·Iₙ block
# This block lives in rows n+1 to n+n and columns n+1 to n+n.
@inbounds for i = 1:n
    rows[nnzh + i] = n + i
    cols[nnzh + i] = n + i
end

# 5. Pre-allocate the augmented right-hand-side vector
b_aug = Vector{T}(undef, n+n)



#testing the solver

# Hx = SparseMatrixCOO(
#       n,#nvar,
#       n, #nvar,
#       rows[1:nnzh], #ls_subsolver.irn[1:ls_subsolver.nnzj],
#       cols[1:nnzh], #ls_subsolver.jcn[1:ls_subsolver.nnzj],
#       vals[1:nnzh], #ls_subsolver.val[1:ls_subsolver.nnzj],
#     )

#1. update the hessian values
hess_coord!(nlp, x, view(vals, 1:nnzh))
σ = 1.0
@inbounds for i = 1:n
    vals[nnzh + i] =  σ
end

b_aug[1:n] .= -1.0 
fill!(view(b_aug, (n + 1):(n + n)), zero(eltype(b_aug)))


Hx = SparseMatrixCOO(
      nnzh + n,#nvar,
      nnzh + n, #nvar,
      rows,
      cols, #ls_subsolver.jcn[1:ls_subsolver.nnzj],
      vals, #ls_subsolver.val[1:ls_subsolver.nnzj],
    )
 H = sparse(cols,rows,vals)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#            TEsting to update the values of H
new_vals = ones(nnzh+n).*10

H.nzval .= new_vals
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 





LBL = Ma97(H)
ma97_factorize!(LBL)
x_sol = ma97_solve(LBL, rhs)  # or x = LBL \ rhs


# # 1. Get problem dimensions and Hessian structure
# meta_nlp = nlp.meta
# nnzh = meta_nlp.nnzh
# n = meta_nlp.nvar


# # 2. Allocate COO arrays for the augmented matrix [H; sqrt(σ)I]
# # Total non-zeros = non-zeros in Hessian (nnzh) + n diagonal entries for the identity block.
# rows = Vector{Int}(undef, nnzh + n)
# cols = Vector{Int}(undef, nnzh + n)
# vals = Vector{T}(undef, nnzh + n)

# # 3. fill in the structure of the Hessian
# hess_structure!(nlp,view(rows, 1:nnzh), view(cols, 1:nnzh))
# hess_coord!(nlp, x, view(vals, 1:nnzh))

# # 4. Fill in the sparsity pattern for the √σ·Iₙ block
# # This block lives in rows n+1 to n+n and columns 1 to n.
# @inbounds for i = 1:n
#     rows[nnzh + i] = n + i
#     cols[nnzh + i] = i
# end


# # 5.Pre-allocate the augmented right-hand-side vector
#     b_aug = Vector{T}(undef, n+n)



# rows, cols = hess_structure(nlp)
# # vals = hess_coord!(nlp, x)