module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf, OrderedCollections

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, SolverParameters

import Krylov.solve!
export solve!

"""
    solve!(solver, nlp)

Solve `nlp` using `solver`.
"""
function solve! end

abstract type AbstractOptSolver{T, V} end

# Unconstrained solvers
include("lbfgs.jl")
include("trunk.jl")

# Unconstrained solvers for NLS
include("trunkls.jl")

# Bound-constrained solvers
include("tron.jl")

# Bound-constrained solvers for NLS
include("tronls.jl")

end
