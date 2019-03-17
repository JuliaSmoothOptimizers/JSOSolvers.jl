# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using Krylov, LinearOperators, NLPModels, SolverTools

# this package
using JSOSolvers

include("consistency.jl")
include("test_solvers.jl")

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
for solver in [lbfgs, tron, trunk]
  @info solver
  solver(nlp, max_eval=20)
  reset!(nlp)
end

nlp = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
for solver in [trunk]
  @info solver
  solver(nlp, max_eval=20)
  reset!(nlp)
end
