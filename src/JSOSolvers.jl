module JSOSolvers

# stdlib
using LinearAlgebra, Logging, Printf

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

"""
    callback_quasi_newton(model, solver, stats)

A default callback for solvers to update the Hessian approximation in quasi-Newton models.
If a user calls a solver with a quasi-Newton model, this will be the default callback.
See https://jso.dev/JSOSolvers.jl/stable/#Callbacks
"""
function callback_quasi_newton(
  model::AbstractNLPModel,
  solver::AbstractSolver,
  stats::GenericExecutionStats,
)
  @debug "in callback_quasi_newton"
  isa(model, NLPModelsModifiers.QuasiNewtonModel) || return
  if !stats.iter_reliable
    @error "iteration counter is not reliable, skipping Hessian approximation update"
    return
  end
  if stats.iter == 0
    # save current gradient for future update
    model.v .= solver.gx
  else
    if !stats.step_status_reliable
      @error "step status is not reliable, skipping Hessian approximation update"
      return
    end
    if stats.step_status == :accepted
      model.v .-= solver.gx
      model.v .*= -1  # v = ∇ₖ₊₁ - ∇ₖ
      push!(model, solver.s, model.v)
      model.v .= solver.gx  # save gradient for next update
    end
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
