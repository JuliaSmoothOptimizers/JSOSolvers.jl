module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

# JSOSolver struct
include("solver-struct.jl")

# Unconstrained solvers
include("lbfgs.jl")
include("trunk.jl")

# Unconstrained solvers for NLS
include("trunkls.jl")

# Bound-constrained solvers
include("tron.jl")

# Tuning
include("tuning/tuning.jl")

end
