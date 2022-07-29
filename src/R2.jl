export R2, R2Solver

"""
    R2(nlp; kwargs...)
    solver = R2Solver(nlp;)
    solve!(solver, nlp; kwargs...)

    A first-order quadratic regularization method for unconstrained optimization

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- x0::V = nlp.meta.x0`: the initial guess
- atol = eps(T)^(1 / 3): absolute tolerance
- rtol = eps(T)^(1 / 3): relative tolerance: algorithm stop when ||∇f(x)|| ≤ ϵ_abs + ϵ_rel*||∇f(x0)||
- η1 = eps(T)^(1/4), η2 = T(0.95): step acceptance parameters
- γ1 = T(1/2), γ2 = 1/γ1: regularization update parameters
- σmin = eps(T): step parameter for R2 algorithm
- max_eval::Int: maximum number of evaluation of the objective function
- max_time::Float64 = 3600.0: maximum time limit in seconds
- verbose::Bool = false: prints iteration details if true.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2(nlp)
# output
"Execution stats: first-order stationary"
```
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = R2Solver(nlp);
stats = solve!(solver, nlp)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct R2Solver{V}
  x::V
  gx::V
  cx::V
end

function R2Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  return R2Solver{V}(x, gx, cx)
end

@doc (@doc R2Solver) function R2(
  nlp::AbstractNLPModel{T,V};
  kwargs...,
) where {T, V}
  solver = R2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function solve!(
    solver::R2Solver{V},
    nlp::AbstractNLPModel{T, V};
    x0::V = nlp.meta.x0,
    atol = eps(T)^(1/3),
    rtol = eps(T)^(1/3),
    η1 = eps(T)^(1/4),
    η2 = T(0.95),
    γ1 = T(1/2),
    γ2 = 1/γ1,
    σmin = zero(T),
    max_time::Float64 = 3600.0,
    max_eval::Int = -1,
    verbose::Bool = true,
  ) where {T, V}

  unconstrained(nlp) || error("R2 should only be called on unconstrained problems.")
  start_time = time()
  elapsed_time = 0.0

  x = solver.x .= x0
  ∇fk = solver.gx
  ck = solver.cx

  iter = 0
  fk = obj(nlp, x)
  
  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  # σk = norm(hess(nlp, x))
  σk = 2^round(log2(norm_∇fk + 1))


  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇fk σk
  end
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  if verbose
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇fk σk
  end

  status = :unknown

  while !(optimal | tired)

    ck .= x .- (∇fk ./ σk)
    ΔTk= norm_∇fk^2 / σk
    fck = obj(nlp, ck)
    if fck == -Inf
      status = :unbounded
      break
    end

    ρk = (fk - fck) / ΔTk 

    # Update regularization parameters
    if ρk >= η2
      σk = max(σmin, γ1 * σk)
    elseif ρk < η1
      σk = σk * γ2
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= ck
      fk = fck
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
    end

    iter += 1
    elapsed_time = time() - start_time
    optimal = norm_∇fk ≤ ϵ
    tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  
    if verbose
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" iter fk norm_∇fk σk
    end

  end
    
  status = if optimal
      :first_order
    elseif tired
      if elapsed_time > max_time
        status = :max_time
      elseif neval_obj(nlp) > max_eval
        status = :max_eval
      end
    end

  return GenericExecutionStats(
      status,
      nlp,
      solution = x,
      objective = fk,
      dual_feas = norm_∇fk,
      elapsed_time = elapsed_time,
      iter = iter,
    )
end