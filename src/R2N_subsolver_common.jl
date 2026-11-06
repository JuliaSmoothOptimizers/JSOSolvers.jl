# ==============================================================================
#   Subsolver Common Interface
#   Defines abstract types and generic functions shared by R2N and R2NLS
# ==============================================================================

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
    update_subsolver!(subsolver::AbstractSubsolver, nlp, x)

Update the internal representation of the subsolver at point `x`.
For R2N solvers, this typically updates the Hessian or Operator.
For R2NLS solvers, this updates the Jacobian.
"""
function update_subsolver! end

"""
    get_model_hessian(subsolver)

Return the model Hessian H (operator or matrix) used by R2N in the
outer loop (curvature, Cauchy point).
"""
function get_model_hessian end

"""
    get_jacobian(subsolver)

Return the operator/matrix J for outer loop calculations in R2NLS.
"""
function get_jacobian end

"""
    get_model_hessian_norm(subsolver)

Return an estimate or upper bound of ‖H‖, the norm of the model Hessian used by R2N.
Used for the Cauchy point.
"""
function get_model_hessian_norm end

"""
    get_jacobian_norm(subsolver)

Return an estimate or upper bound of ‖J‖, the norm of the Jacobian returned by
[`get_jacobian`](@ref). Used by R2NLS for the Cauchy point.
"""
function get_jacobian_norm end

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
    is_unsupported(subsolver)

Return `true` if the subsolver cannot be used for the current problem (e.g. the
Hessian is too dense for a direct sparse factorization). When `true`, the outer
solver should skip solving and return with status `:exception`. Defaults to
`false`.
"""
is_unsupported(sub::AbstractSubsolver) = false

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

"""
    finalize_subsolver!(subsolver::AbstractSubsolver)

Explicitly free any C/Fortran memory allocated by the subsolver to prevent leaks.
"""
finalize_subsolver!(subsolver::AbstractSubsolver) = nothing

"""
    reset_subsolver!(subsolver::AbstractSubsolver, nlp, x)

Reset the subsolver state. Default behavior is to finalize and re-initialize.
"""
function reset_subsolver!(subsolver::AbstractSubsolver, nlp, x)
  finalize_subsolver!(subsolver)
  initialize!(subsolver, nlp, x)
  return subsolver
end
