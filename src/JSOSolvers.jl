module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

import SolverCore.solve!
import Krylov.solve!
export solve!

# Unconstrained solvers
include("lbfgs.jl")
include("trunk.jl")
include("R2.jl")

# Unconstrained solvers for NLS
include("trunkls.jl")

# List of keywords accepted by TRONTrustRegion
const tron_keys = (
  :max_radius,
  :acceptance_threshold,
  :decrease_threshold,
  :increase_threshold,
  :large_decrease_factor,
  :small_decrease_factor,
  :increase_factor,
)

# Bound-constrained solvers
include("tron.jl")

# Bound-constrained solvers for NLS
include("tronls.jl")

end
