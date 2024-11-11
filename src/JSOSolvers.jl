module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov,
  LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverParameters, SolverTools

import SolverCore.solve!
export solve!

const Callback_docstring = "
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of the residual, for instance, the norm of the gradient for unconstrained problems;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
"

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

# Unconstrained solvers
include("lbfgs.jl")
include("trunk.jl")
include("fomo.jl")

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
