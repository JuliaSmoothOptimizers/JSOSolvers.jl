export LBFGSSolver

mutable struct LBFGSSolver{T, S} <: AbstractOptSolver{T, S}
  initialized::Bool
  params::Dict
  workspace
end

function SolverCore.parameters(::Type{LBFGSSolver{T, S}}) where {T, S}
  (
    mem = (default = 5, type = Int, min = 1, max = 10),
  )
end

function SolverCore.are_valid_parameters(::Type{LBFGSSolver}, mem)
  return mem ≥ 1
end

"""
    LBFGSSolver(nlp)

An implementation of a limited memory BFGS line-search method for unconstrained
minimization.
"""
function LBFGSSolver{T, S}(
  meta::AbstractNLPModelMeta;
  x0::S = meta.x0,
  kwargs...,
) where {T, S}
  nvar, ncon = meta.nvar, meta.ncon
  params = parameters(LBFGSSolver{T, S})
  solver = LBFGSSolver{T, S}(
    true,
    Dict(k => v[:default] for (k, v) in pairs(params)),
    ( # workspace
      x = S(undef, nvar),
    ),
  )
  for (k, v) in kwargs
    solver.params[k] = v
  end
  solver
end

function SolverCore.solve!(
  solver :: LBFGSSolver{T, S},
  nlp :: AbstractNLPModel;
  x0 :: S=nlp.meta.x0,
  atol :: T=√eps(T),
  rtol :: T=√eps(T),
  max_eval :: Int=-1,
  max_time :: Float64=30.0,
  verbose :: Bool=true,
  kwargs...
) where {T, S}

  if !unconstrained(nlp)
    error("lbfgs should only be called for unconstrained problems. Try tron instead")
  end

  start_time = time()
  elapsed_time = 0.0

  mem = solver.params[:mem]

  n = nlp.meta.nvar

  x = solver.workspace.x .= x0
  xt = zeros(T, n)
  ∇ft = zeros(T, n)

  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  H = InverseLBFGSOperator(T, n, mem, scaling=true)

  ∇fNorm = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm
  iter = 0

  @info log_header([:iter, :f, :dual, :slope, :bk], [Int, T, T, T, Int],
                   hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))

  optimal = ∇fNorm ≤ ϵ
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  stalled = false
  status = :unknown

  h = LineModel(nlp, x, ∇f)

  while !(optimal || tired || stalled)
    d = - H * ∇f
    slope = dot(n, d, ∇f)
    if slope ≥ 0
      @error "not a descent direction" slope
      status = :not_desc
      stalled = true
      continue
    end

    redirect!(h, x, d)
    # Perform improved Armijo linesearch.
    t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=T(0.9999), bk_max=25, verbose=false)

    @info log_row(Any[iter, f, ∇fNorm, slope, nbk])

    copyaxpy!(n, t, d, x, xt)
    good_grad || grad!(nlp, xt, ∇ft)

    # Update L-BFGS approximation.
    push!(H, t * d, ∇ft - ∇f)

    # Move on.
    x .= xt
    f = ft
    ∇f .= ∇ft

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

  return OptSolverOutput(status, x, nlp, objective=f, dual_feas=∇fNorm,
                         iter=iter, elapsed_time=elapsed_time)
end
