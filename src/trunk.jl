export trunk

trunk(nlp::AbstractNLPModel; variant = :Newton, kwargs...) = trunk(Val(variant), nlp; kwargs...)

"""
    trunk(nlp)

A trust-region solver for unconstrained optimization using exact second derivatives.

This implementation follows the description given in [1].
The main algorithm follows the basic trust-region method described in Section 6.
The backtracking linesearch follows Section 10.3.2.
The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.

[1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint,
    Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.
    SIAM, Philadelphia, USA, 2000.
    DOI: 10.1137/1.9780898719857.
"""
function trunk(
  ::Val{:Newton},
  nlp::AbstractNLPModel;
  subsolver_logger::AbstractLogger = NullLogger(),
  x::AbstractVector = copy(nlp.meta.x0),
  atol::Real = √eps(eltype(x)),
  rtol::Real = √eps(eltype(x)),
  max_eval::Int = -1,
  max_time::Float64 = 30.0,
  bk_max::Int = 10,
  monotone::Bool = true,
  nm_itmax::Int = 25,
)
  if !(nlp.meta.minimize)
    error("trunk only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("trunk should only be called for unconstrained problems. Try tron instead")
  end

  start_time = time()
  elapsed_time = 0.0

  T = eltype(x)
  n = nlp.meta.nvar

  cgtol = one(T)  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = eps(T)^T(1 / 4)

  iter = 0
  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  gt = zeros(T, n)
  ∇fNorm2 = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm2
  tr = TrustRegion(gt, min(max(∇fNorm2 / 10, one(T)), T(100)))

  # Non-monotone mode parameters.
  # fmin: current best overall objective value
  # nm_iter: number of successful iterations since fmin was first attained
  # fref: objective value at reference iteration
  # σref: cumulative model decrease over successful iterations since the reference iteration
  fmin = fref = fcur = f
  σref = σcur = zero(T)
  nm_iter = 0

  # Preallocate xt.
  xt = Vector{T}(undef, n)
  temp = Vector{T}(undef, n)

  optimal = ∇fNorm2 ≤ ϵ
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  stalled = false
  status = :unknown
  solved = optimal || tired || stalled

  if isa(nlp, QuasiNewtonModel) && !solved
    ∇fn = copy(∇f)
  end

  @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :inner, :bk, :cgstatus],
    [Int, T, T, T, T, Int, Int, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )

  while !(solved || tired || stalled)
    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    H = hess_op!(nlp, x, temp)
    cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
    (s, cg_stats) = with_logger(subsolver_logger) do
      cg(H, -∇f, atol = T(atol), rtol = cgtol, radius = tr.radius, itmax = max(2 * n, 50))
    end

    # Compute actual vs. predicted reduction.
    sNorm = nrm2(n, s)
    copyaxpy!(n, one(T), s, x, xt)
    slope = dot(n, s, ∇f)
    curv = dot(n, s, H * s)
    Δq = slope + curv / 2
    ft = obj(nlp, xt)

    ared, pred = aredpred!(tr, nlp, f, ft, Δq, xt, s, slope)
    if pred ≥ 0
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
      if pred_hist ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      ρ_hist = ared_hist / pred_hist
      tr.ratio = max(tr.ratio, ρ_hist)
    end

    bk = 0
    if !acceptable(tr)
      # Perform backtracking linesearch along s
      # Scaling s to the trust-region boundary, as recommended in
      # Algorithm 10.3.2 of the Trust-Region book
      # appears to deteriorate results.
      # BLAS.scal!(n, tr.radius / sNorm, s, 1)
      # slope *= tr.radius / sNorm
      # sNorm = tr.radius

      if slope ≥ 0
        @error "not a descent direction: slope = $slope, ‖∇f‖ = $∇fNorm2"
        status = :not_desc
        stalled = true
        continue
      end
      α = one(T)
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= T(1.2)
        copyaxpy!(n, α, s, x, xt)
        ft = obj(nlp, xt)
      end
      sNorm *= α
      scal!(n, α, s)
      slope *= α
      Δq = slope + α * α * curv / 2
      ared, pred = aredpred!(tr, nlp, f, ft, Δq, xt, s, slope)
      if pred ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = aredpred!(tr, nlp, fref, ft, σref + Δq, xt, s, slope)
        if pred_hist ≥ 0
          status = :neg_pred
          stalled = true
          continue
        end
        ρ_hist = ared_hist / pred_hist
        tr.ratio = max(tr.ratio, ρ_hist)
      end
    end

    @info log_row([
      iter,
      f,
      ∇fNorm2,
      tr.radius,
      tr.ratio,
      length(cg_stats.residuals),
      bk,
      cg_stats.status,
    ])
    iter = iter + 1

    if acceptable(tr)
      # Update non-monotone mode parameters.
      if !monotone
        σref = σref + Δq
        σcur = σcur + Δq
        if ft < fmin
          # New overall best objective value found.
          fcur = ft
          fmin = ft
          σcur = zero(T)
          nm_iter = 0
        else
          nm_iter = nm_iter + 1

          if ft > fcur
            fcur = ft
            σcur = zero(T)
          end

          if nm_iter ≥ nm_itmax
            fref = fcur
            σref = σcur
          end
        end
      end

      x .= xt
      f = ft
      if !tr.good_grad
        grad!(nlp, x, ∇f)
        tr.good_grad = false
      else
        ∇f .= tr.gt
      end
      ∇fNorm2 = nrm2(n, ∇f)

      if isa(nlp, QuasiNewtonModel)
        ∇fn .-= ∇f
        ∇fn .*= -1  # = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, ∇fn)
        ∇fn .= ∇f
      end
    end

    # Move on.
    update!(tr, sNorm)

    optimal = ∇fNorm2 ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time
    solved = optimal || tired || stalled
  end
  @info log_row(Any[iter, f, ∇fNorm2, tr.radius])

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_eval ≥ 0
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = f,
    dual_feas = ∇fNorm2,
    iter = iter,
    elapsed_time = elapsed_time,
  )
end
