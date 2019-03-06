# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using Krylov, LinearOperators, NLPModels, SolverTools

@static Sys.isunix() && using CUTEst

# this package
using JSOSolvers

include("consistency.jl")

include("test_solvers.jl")
