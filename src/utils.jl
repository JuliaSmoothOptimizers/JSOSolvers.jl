# — Power‐method for square B to get ∥B∥₂ = largest |eigenvalue| —
import LinearAlgebra: opnorm       # bring the Base name into your namespace
export opnorm  

function opnorm_power_square(B; tol=eps(BigFloat), maxiter=1000)
    n = size(B,2)
    x = rand(BigFloat, n); x /= norm(x)
    λ_old = zero(BigFloat)
    for _ in 1:maxiter
        y = B*x
        λ = norm(y)
        x = y/λ
        if abs(λ - λ_old) < tol*λ
            return λ, true
        end
        λ_old = λ
    end
    return λ_old, false
end

# — Power‐method for rectangular J to get ∥J∥₂ = √(max eig(B'B)) —
function opnorm_power_rect(J; tol=eps(BigFloat), maxiter=1000)
    m, n = size(J)
    x = rand(BigFloat, n); x /= norm(x)
    σ_old = zero(BigFloat)
    for _ in 1:maxiter
        y = J * x              # in ℝ^m
        z = J' * y             # back in ℝ^n
        σ = norm(y)            # candidate singular‐value
        x = z / norm(z)        # next direction
        if abs(σ - σ_old) < tol*σ
            return σ, true
        end
        σ_old = σ
    end
    return σ_old, false
end

# — Top‐level dispatch —
function LinearAlgebra.opnorm(B; kwargs...)
    m, n = size(B)
    return (m == n ? opnorm_eig : opnorm_svd)(B; kwargs...)
end

function opnorm_eig(B; max_attempts::Int = 3)

  # 1) BigFloat: pure‐Julia power‐method
  if eltype(B) === BigFloat
      return opnorm_power_square(B)
  end

  n = size(B,1)
  # 2) tiny dense Float64: direct LAPACK
  if n ≤ 5
      return maximum(abs, eigen(Matrix(B)).values), true
  end

  # 3) iterative ARPACK
  nev, ncv = 1, max(20, 2*1 + 1)
  attempt, λ, have_eig = 0, zero(eltype(B)), false

  while !(have_eig || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest eigenvalue in absolute value
      d, nconv, niter, nmult, resid =
        eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)

      # Check if eigenvalue has converged
      have_eig = nconv == 1
      if have_eig
        λ = abs(d[1])  # Take absolute value of the largest eigenvalue
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return λ, have_eig
end

function opnorm_svd(J; max_attempts::Int = 3)
    # 1) BigFloat: pure‐Julia rectangular power‐method
  if eltype(J) === BigFloat
      return opnorm_power_rect(J)
  end

  m, n = size(J)
  # 2) tiny dense Float64: direct LAPACK
  if min(m,n) ≤ 5
      return maximum(svd(Matrix(J)).S), true
  end

  # 3) iterative ARPACK‐SVD
  nsv, ncv = 1, 10
  attempt, σ, have_svd = 0, zero(eltype(J)), false
  n = min(m,n)

  while !(have_svd || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest singular value
      s, nconv, niter, nmult, resid = svds(J; nsv = nsv, ncv = ncv, ritzvec = false, check = 1)

      # Check if singular value has converged
      have_svd = nconv >= 1
      if have_svd
        σ = maximum(s.S)  # Take the largest singular value
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return σ, have_svd
end
