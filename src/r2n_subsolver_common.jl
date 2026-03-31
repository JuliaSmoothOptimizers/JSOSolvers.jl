# ==============================================================================
#   Subsolver Common Interface
#   Defines abstract types and generic functions shared by R2N and R2NLS
# ==============================================================================

export AbstractSubsolver, AbstractR2NSubsolver, AbstractR2NLSSubsolver
export initialize!, update_subsolver!
export get_operator,
  get_jacobian, get_inertia, has_npc_direction, get_npc_direction, get_operator_norm

"""
    AbstractSubsolver{T}

Base abstract type for all subproblem solvers.

# Interface
Subtypes of `AbstractSubsolver` must be callable (act as functors) to solve the subproblem:
    (subsolver::AbstractSubsolver)(s, rhs, σ, atol, rtol; verbose=0)
"""
abstract type AbstractSubsolver{T} end

"""
    AbstractR2NSubsolver{T}

Abstract type for subsolvers used in the R2N (Newton-type) algorithm.
"""
abstract type AbstractR2NSubsolver{T} <: AbstractSubsolver{T} end

"""
    AbstractR2NLSSubsolver{T}

Abstract type for subsolvers used in the R2NLS (Nonlinear Least Squares) algorithm.
"""
abstract type AbstractR2NLSSubsolver{T} <: AbstractSubsolver{T} end

# ==============================================================================
#   Generic Functions (Interface)
# ==============================================================================

"""
    initialize!(subsolver::AbstractSubsolver, args...)

Perform initial setup for the subsolver (e.g., analyze sparsity, allocate workspace).
This is typically called once at the start of the optimization.
"""
function initialize! end

"""
    update_subsolver!(subsolver::AbstractSubsolver, model, x)

Update the internal representation of the subsolver at point `x`.
For R2N solvers, this typically updates the Hessian or Operator.
For R2NLS solvers, this updates the Jacobian.
"""
function update_subsolver! end

"""
    get_operator(subsolver)

Return the operator/matrix H for outer loop calculations (curvature, Cauchy point) in R2N.
"""
function get_operator end

"""
    get_jacobian(subsolver)

Return the operator/matrix J for outer loop calculations in R2NLS.
"""
function get_jacobian end

"""
    get_operator_norm(subsolver)

Return the norm (usually infinity norm or estimate) of the operator H or J.
Used for Cauchy point calculation.
"""
function get_operator_norm end

# ==============================================================================
#   Optional Interface Methods (Default Implementations)
# ==============================================================================

"""
    get_inertia(subsolver)

Return (num_neg, num_zero) eigenvalues of the underlying operator.
Returns (-1, -1) if unknown or not applicable.
"""
function get_inertia(sub)
  return -1, -1
end

"""
    has_npc_direction(subsolver)

Return `true` if a direction of negative curvature was found during the last solve, `false` otherwise.
"""
has_npc_direction(sub) = false

"""
    get_npc_direction(subsolver)

Return a direction of negative curvature. Raises an error if none was found or if the subsolver does not support it.
"""
get_npc_direction(sub) = error("No NPC direction available for $(typeof(sub)).")
