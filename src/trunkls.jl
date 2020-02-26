const trunkls_allowed_subsolvers = [:cgls, :crls, :lsqr, :lsmr]

"""
    trunk(nls)

A trust-region solver for nonlinear least squares.

This implementation follows the description given in [1].
The main algorithm follows the basic trust-region method described in Section 6.
The backtracking linesearch follows Section 10.3.2.
The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.

[1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint,
    Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.
    SIAM, Philadelphia, USA, 2000.
    DOI: 10.1137/1.9780898719857.
"""
function trunk(nlp :: AbstractNLSModel;
               x :: AbstractVector=copy(nlp.meta.x0),
               subsolver :: Symbol=:lsmr,
               atol :: Real=√eps(eltype(x)), rtol :: Real=√eps(eltype(x)),
               max_eval :: Int=-1,
               max_time :: Float64=30.0,
               bk_max :: Int=10,
               monotone :: Bool=true,
               nm_itmax :: Int=25,
               trsolver_args :: Dict{Symbol,Any}=Dict{Symbol,Any}())

  start_time = time()
  elapsed_time = 0.0

  subsolver in trunkls_allowed_subsolvers || error("subproblem solver must be one of $(trunkls_allowed_subsolvers)")
  trsolver = eval(subsolver)
  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ
  T = eltype(x)
  cgtol = one(T)  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = eps(T)^T(1/4)

  iter = 0
  r = residual(nlp, x)
  f = dot(r, r) / 2

  # preallocate storage for products with A and A'
  Av = Vector{T}(undef, m)
  Atv = Vector{T}(undef, n)
  A = jac_op_residual!(nlp, x, Av, Atv)
  ∇f = A' * r
  ∇fNorm2 = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm2
  tr = TrustRegion(min(max(∇fNorm2 / 10, one(T)), T(100)))

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
  tired = neval_residual(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  stalled = false
  status = :unknown

  @info log_header([:iter, :f, :dual, :radius, :step, :ratio, :inner, :bk, :cgstatus], [Int, T, T, T, T, T, Int, Int, String],
                   hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :radius=>"Δ"))

  while !(optimal || tired || stalled)
    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    cgtol = max(rtol, min(T(0.1), 9 * cgtol / 10, sqrt(∇fNorm2)))
    (s, cg_stats) = trsolver(A, -r,
                             atol=atol, rtol=cgtol,
                             radius=get_property(tr, :radius),
                             itmax=max(2 * (n + m), 50), verbose=false;
                             trsolver_args...)

    # Compute actual vs. predicted reduction.
    sNorm = nrm2(n, s)
    copyaxpy!(n, one(T), s, x, xt)
    # slope = dot(∇f, s)
    t = A * s
    slope = dot(r, t)
    curv = dot(t, t)
    Δq = slope + curv / 2
    rt = residual(nlp, xt)
    ft = dot(rt, rt) / 2
    @debug @sprintf("‖s‖ = %7.1e, slope = %8.1e, Δq = %15.7e", sNorm, slope, Δq)

    ared, pred = aredpred(nlp, f, ft, Δq, xt, s, slope)
    if pred ≥ 0
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred(nlp, fref, ft, σref + Δq, xt, s, slope)
      if pred_hist ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      ρ_hist = ared_hist / pred_hist
      set_property!(tr, :ratio, max(get_property(tr, :ratio), ρ_hist))
    end

    bk = 0
    if !acceptable(tr)
      # Perform backtracking linesearch along s
      # Scaling s to the trust-region boundary, as recommended in
      # Algorithm 10.3.2 of the Trust-Region book
      # appears to deteriorate results.
      # BLAS.scal!(n, get_property(tr, :radius) / sNorm, s, 1)
      # slope *= get_property(tr, :radius) / sNorm
      # sNorm = get_property(tr, :radius)

      if slope ≥ 0
        @error "not a descent direction" slope ∇fNorm2 sNorm
        status = :not_desc
        stalled = true
        continue
      end
      α = one(T)
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= T(1.2)
        copyaxpy!(n, α, s, x, xt)
        rt = residual(nlp, xt)
        ft = dot(rt, rt) / 2
        @debug "" α ft
      end
      sNorm *= α
      scal!(n, α, s)
      slope *= α
      Δq = slope + α * α * curv / 2
      @debug "" slope Δq
      ared, pred = aredpred(nlp, f, ft, Δq, xt, s, slope)
      if pred ≥ 0
        status = :neg_pred
        stalled = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = ratio(nlp, fref, ft, σref + Δq, xt, s, slope)
        if pred_hist ≥ 0
          status = :neg_pred
          stalled = true
          continue
        end
        ρ_hist = ared_hist / pred_hist
        set_property!(tr, :ratio, max(get_property(tr, :ratio), ρ_hist))
      end
    end

    @info log_row([iter, f, ∇fNorm2, get_property(tr, :radius), sNorm,
                   get_property(tr, :ratio), length(cg_stats.residuals), bk, cg_stats.status])
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

          if nm_iter >= nm_itmax
            fref = fcur
            σref = σcur
          end
        end
      end

      x .= xt
      r = rt
      f = ft
      A = jac_op_residual!(nlp, x, Av, Atv)
      ∇f = A' * r
      ∇fNorm2 = nrm2(n, ∇f)
    end

    # Move on.
    update!(tr, sNorm)

    optimal = ∇fNorm2 ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_residual(nlp) > max_eval ≥ 0 || elapsed_time > max_time
  end
  @info log_row(Any[iter, f, ∇fNorm2, get_property(tr, :radius)])

  if optimal
    status = :first_order
  elseif tired
    if neval_residual(nlp) > max_eval ≥ 0
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=f, dual_feas=∇fNorm2,
                               iter=iter, elapsed_time=elapsed_time)
end
