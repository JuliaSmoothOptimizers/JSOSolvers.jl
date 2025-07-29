import LinearAlgebra: opnorm       # bring the Base name into your namespace
using GenericLinearAlgebra
using TSVD

export opnorm
# — Top‐level dispatch —
function LinearAlgebra.opnorm(B; kwargs...)
  _opnorm(B, eltype(B); kwargs...)
end

# This method will be picked if eltype is one of the four types Arpack supports
# (Float32, Float64, ComplexF32, ComplexF64).
function _opnorm(
  B,
  ::Type{T};
  kwargs...,
) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
  m, n = size(B)
  return (m == n ? opnorm_eig : opnorm_svd)(B; kwargs...)
end

# Fallback for everything else
function _opnorm(B, ::Type{T}; kwargs...) where {T}
  _, s, _ = tsvd(B)
  return s[1], true  # return largest singular value
end

function opnorm_eig(B; max_attempts::Int = 3)
  n = size(B, 1)
  # 1) tiny dense Float64: direct LAPACK
  if n ≤ 5
    return maximum(abs, eigen(Matrix(B)).values), true
  end

  # 2) iterative ARPACK
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
  m, n = size(J)
  # 1) tiny dense Float64: direct LAPACK
  if min(m, n) ≤ 5
    return maximum(svd(Matrix(J)).S), true
  end

  # 2) iterative ARPACK‐SVD
  nsv, ncv = 1, 10
  attempt, σ, have_svd = 0, zero(eltype(J)), false
  n = min(m, n)

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