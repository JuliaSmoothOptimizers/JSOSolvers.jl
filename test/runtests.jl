# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, OptSolver, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

# this package
using JSOSolvers

include("consistency.jl")
include("test_solvers.jl")

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
for Solver in [LBFGSSolver, TronSolver, TrunkSolver]
  @info Solver
  solver = Solver(nlp)
  solve!(solver, nlp, max_eval=20)
  reset!(nlp)
end

nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
for Solver in [TrunkNLSSolver]
  @info Solver
  solver = Solver(nls)
  solve!(solver, nls, max_eval=20)
  reset!(nls)
end

include("objgrad-on-tron.jl")