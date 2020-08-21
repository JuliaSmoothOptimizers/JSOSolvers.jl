export lbfgs

"""
    lbfgs(nlp)

An implementation of a limited memory BFGS line-search method for unconstrained
minimization.
"""
function lbfgs(nlp :: AbstractNLPModel;
               x :: AbstractVector=copy(nlp.meta.x0),
               atol :: Real=√eps(eltype(x)), rtol :: Real=√eps(eltype(x)),
               max_eval :: Int=-1,
               max_time :: Float64=30.0,
               ls_method :: Symbol=:armijo_wolfe,
               mem :: Int=5)

  if !unconstrained(nlp)
    error("lbfgs should only be called for unconstrained problems. Try tron instead")
  end

  start_time = time()
  elapsed_time = 0.0

  T = eltype(x)
  n = nlp.meta.nvar

  xt = zeros(T, n)
  ∇fold = zeros(T, n)

  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  H = InverseLBFGSOperator(T, n, mem, scaling=true)
  ϕ = UncMerit(nlp, fx=f, gx=∇f)

  ∇fNorm = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm
  iter = 0

  @info log_header([:iter, :f, :dual, :slope, :bk], [Int, T, T, T, Int],
                   hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))

  optimal = ∇fNorm ≤ ϵ
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  stalled = false
  status = :unknown

  while !(optimal || tired || stalled)
    d = - H * ∇f
    ∇fold .= ∇f
    # Perform improved Armijo linesearch.
    lso = linesearch!(ϕ, x, d, xt, method=ls_method, bk_max=25)

    @info log_row(Any[iter, f, ∇fNorm, dot(d, ∇fold), lso.specific[:nbk]])

    lso.good_grad || grad!(nlp, xt, ∇f)

    # Update L-BFGS approximation.
    push!(H, lso.t * d, ∇f - ∇fold)

    # Move on.
    x .= xt
    ϕ.fx = f = lso.ϕt

    ∇fNorm = nrm2(n, ∇f)
    iter = iter + 1

    optimal = ∇fNorm ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  end
  @info log_row(Any[iter, f, ∇fNorm])

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_eval ≥ 0
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=f, dual_feas=∇fNorm,
                               iter=iter, elapsed_time=elapsed_time)
end
