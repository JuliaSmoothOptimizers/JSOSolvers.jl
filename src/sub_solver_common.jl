# ==============================================================================
#   Subsolver Common Interface
#   Defines abstract types and generic functions shared by R2N and R2NLS
# ==============================================================================

export AbstractR2NSubsolver, AbstractR2NLSSubsolver
export initialize_subsolver!, update_subsolver!, update_jacobian!
export solve_subproblem!
export get_operator, get_jacobian, get_inertia, get_npc_direction, get_operator_norm

"""
    AbstractR2NSubsolver{T}

Abstract type for subsolvers used in the R2N (Newton-type) algorithm.
"""
abstract type AbstractR2NSubsolver{T} end

"""
    AbstractR2NLSSubsolver{T}

Abstract type for subsolvers used in the R2NLS (Nonlinear Least Squares) algorithm.
"""
abstract type AbstractR2NLSSubsolver{T} end

# ==============================================================================
#   Generic Functions (Interface)
# ==============================================================================

"""
    initialize_subsolver!(subsolver, nlp, x)

Perform initial setup for the subsolver (e.g., analyze sparsity, allocate workspace).
This is typically called once at the start of the optimization.
"""
function initialize_subsolver! end

"""
    update_subsolver!(subsolver, nlp, x)

Update the internal Hessian or Operator representation at point `x` for R2N solvers.
"""
function update_subsolver! end

"""
    update_jacobian!(subsolver, nls, x)

Update the internal Jacobian representation at point `x` for R2NLS solvers.
"""
function update_jacobian! end

"""
    solve_subproblem!(subsolver, s, rhs, σ, atol, rtol; verbose=0)

Solve the regularized subproblem:
- For R2N:   (H + σI)s = rhs
- For R2NLS: min ||Js - rhs||² + σ||s||²  (or equivalent normal equations)

Returns: (solved::Bool, status::Symbol, niter::Int) or (solved, status, niter, npc)
"""
function solve_subproblem! end

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
    get_npc_direction(subsolver)

Return a direction of negative curvature if one was found during the solve.
Returns `nothing` or `sub.x` if not found.
"""
function get_npc_direction(sub)
  # Default fallback; specific solvers should override if they support NPC detection
  return nothing
end