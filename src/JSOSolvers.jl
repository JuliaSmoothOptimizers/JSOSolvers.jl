module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

import Krylov.solve!
export solve!

"""
    solve!(solver, nlp)

Solve `nlp` using `solver`.
"""
function solve! end

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
