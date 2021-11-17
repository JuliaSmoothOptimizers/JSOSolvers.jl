using BenchmarkTools
# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverBenchmark, SolverCore, SolverTools, ADNLPModels, SolverTest, JSOSolvers, CUTEst

function runcutest(cutest_problems, solvers)
  return bmark_solvers(solvers, cutest_problems)
end


problems_names = CUTEst.select(;only_free_var=true, max_con=0, min_var=2, max_var=100)
cutest_problems = (CUTEstModel(p) for p in problems_names)
solvers = Dict(
  :lbfgs =>
  nlp -> lbfgs(nlp)
  )
  
const SUITE = BenchmarkGroup()
SUITE[:cutest_lbfgs] = @benchmarkable runcutest(cutest_problems, solvers)
tune!(SUITE[:cutest_lbfgs])
