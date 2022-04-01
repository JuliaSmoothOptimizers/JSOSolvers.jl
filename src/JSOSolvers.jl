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

abstract type AbstractOptSolver{T, V} end

# TODO: Move to SolverCore
function get_status(
  nlp;
  elapsed_time = 0.0,
  optimal = false,
  max_eval = Inf,
  max_time = Inf,
)
  if optimal
    :first_order
  elseif neval_obj(nlp) > max_eval ≥ 0
    :max_eval
  elseif elapsed_time > max_time
    :max_time
  else
    :unknown
  end
end

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
