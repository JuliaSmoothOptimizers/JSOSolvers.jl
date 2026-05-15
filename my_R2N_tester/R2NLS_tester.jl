using Revise
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


stats_QRmumps = R2NLS(nls, subsolver=QRMumpsSubsolver, verbose=1, max_iter=100)

print("running R2NLS with LSMR subsolver... ")
stats_LSMR = R2NLS(nls, subsolver=LSMRSubsolver, verbose=1, max_iter=100)



# Trunkls
stats_trunkls = trunk(nls, verbose=1, max_iter=100)


println("\nStastics Summary:")
println("stats_QRmumps: ", stats_QRmumps)
println("stats_LSMR: ", stats_LSMR)
println("stats_trunkls: ", stats_trunkls)
