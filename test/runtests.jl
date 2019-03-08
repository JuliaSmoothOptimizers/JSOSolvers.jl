# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using Krylov, LinearOperators, NLPModels, SolverTools

# this package
using JSOSolvers

include("consistency.jl")

include("test_solvers.jl")
