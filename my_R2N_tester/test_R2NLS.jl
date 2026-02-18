using JSOSolvers
using ADNLPModels
using SolverCore
using LinearAlgebra
using SparseArrays
using Printf
using QRMumps
using Krylov
using LinearOperators

# 1. Define the Rosenbrock Problem
# Residual F(x) = [10(x2 - x1^2); 1 - x1]
# Minimum at [1, 1]
rosenbrock_f(x) = [10 * (x[2] - x[1]^2); 1 - x[1]]
nls = ADNLSModel(rosenbrock_f, [-1.2; 1.0], 2, name="Rosenbrock")

println("Problem: $(nls.meta.name)")
println("Initial x: $(nls.meta.x0)")

# 2. Run R2NLS with default settings (QRMumps subsolver)
println("\nRunning R2NLS...")
stats = R2NLS(nls,max_iter = 100 ,verbose=1)

 
########### Additional tests can be added here, e.g., with different subsolvers or on different problems

stats = R2NLS(nls, subsolver=LSMRSubsolver, verbose=1)

 