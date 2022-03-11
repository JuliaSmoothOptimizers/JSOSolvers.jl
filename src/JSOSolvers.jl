module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, SolverParameters

import Krylov.solve!
export solve!, get_parameters

"""
    solve!(solver, nlp)

Solve `nlp` using `solver`.
"""
function solve! end

abstract type AbstractOptSolver{T, V} end

"""
Returns the set of parameters of a solver as a Dict.
"""
function get_parameters(::AbstractOptSolver) end

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
