# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

# this package
using JSOSolvers

include("simple_model.jl")
include("consistency.jl")
include("test_solvers.jl")
if VERSION ≥ v"1.6"
  include("allocs.jl")
end

include("objgrad-on-tron.jl")
