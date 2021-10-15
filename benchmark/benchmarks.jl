using BenchmarkTools
# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, ADNLPModels, SolverTest, JSOSolvers
const SUITE = BenchmarkGroup()
SUITE[:lbfgs] = @benchmarkable unconstrained_nlp(nlp -> lbfgs(nlp))