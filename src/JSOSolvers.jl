module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using Krylov,
  LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverParameters, SolverTools

import SolverTools.reset!
import SolverCore.solve!
export solve!

"""
    normM!(n, x, M, z)
    
Weighted norm of `x` with respect to `M`, i.e., `z = sqrt(x' * M * x)`. Uses `z` as workspace.
"""
function normM!(n, x, M, z)
  if M === I
    return nrm2(n, x)
  else
    mul!(z, M, x)
    return √(x ⋅ z)
  end
end

# subsolver interface
include("sub_solver_common.jl")

# Unconstrained solvers
include("lbfgs.jl")
include("trunk.jl")
include("fomo.jl")
include("R2N.jl")

# Unconstrained solvers for NLS
include("trunkls.jl")
include("R2Nls.jl")

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
