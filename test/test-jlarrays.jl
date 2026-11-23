# GPU-array regression tests that run without a GPU. JLArrays is a CPU-backed GPU array type
# that forbids scalar indexing like CUDA does. The problems below only use broadcasts and
# reductions, so any scalar indexing that remains comes from the solvers themselves
# (NLPModelsTest problems index elementwise, which is why test-gpu.jl needs `allowscalar`).

using JLArrays
JLArrays.allowscalar(false)

# f(x) = ½‖x - 1‖² + ¼‖x.^2 - 1‖²: convex, unique minimizer x = 1.
mutable struct BroadcastQuartic{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end
function BroadcastQuartic(::Type{S}, n::Int = 10) where {S}
  T = eltype(S)
  x0 = fill!(S(undef, n), T(2))
  meta = NLPModelMeta{T, S}(n; x0 = x0, lvar = fill!(S(undef, n), T(-Inf)),
                            uvar = fill!(S(undef, n), T(Inf)), variable_bounds_analysis = false)
  return BroadcastQuartic(meta, Counters())
end
function NLPModels.obj(nlp::BroadcastQuartic, x::AbstractVector)
  increment!(nlp, :neval_obj)
  return sum(abs2, x .- 1) / 2 + sum(abs2, x .^ 2 .- 1) / 4
end
function NLPModels.grad!(nlp::BroadcastQuartic, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  g .= (x .- 1) .+ (x .^ 2 .- 1) .* x
  return g
end

# Same objective as ½‖F(x)‖² with F(x) = [x - 1; (x.^2 - 1)/√2].
mutable struct BroadcastQuarticNLS{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end
function BroadcastQuarticNLS(::Type{S}, n::Int = 10) where {S}
  T = eltype(S)
  x0 = fill!(S(undef, n), T(2))
  meta = NLPModelMeta{T, S}(n; x0 = x0, lvar = fill!(S(undef, n), T(-Inf)),
                            uvar = fill!(S(undef, n), T(Inf)), variable_bounds_analysis = false)
  return BroadcastQuarticNLS(meta, NLSMeta{T, S}(2n, n; x0 = x0), NLSCounters())
end
function NLPModels.residual!(nls::BroadcastQuarticNLS, x::AbstractVector, Fx::AbstractVector)
  increment!(nls, :neval_residual)
  n = nls.meta.nvar
  view(Fx, 1:n) .= x .- 1
  view(Fx, (n + 1):(2n)) .= (x .^ 2 .- 1) ./ sqrt(eltype(x)(2))
  return Fx
end
function NLPModels.jprod_residual!(
  nls::BroadcastQuarticNLS,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nls, :neval_jprod_residual)
  n = nls.meta.nvar
  view(Jv, 1:n) .= v
  view(Jv, (n + 1):(2n)) .= sqrt(eltype(x)(2)) .* x .* v
  return Jv
end
function NLPModels.jtprod_residual!(
  nls::BroadcastQuarticNLS,
  x::AbstractVector,
  w::AbstractVector,
  Jtw::AbstractVector,
)
  increment!(nls, :neval_jtprod_residual)
  n = nls.meta.nvar
  Jtw .= view(w, 1:n) .+ sqrt(eltype(x)(2)) .* x .* view(w, (n + 1):(2n))
  return Jtw
end

@testset "JLArray, scalar indexing disallowed" begin
  S = JLArray{Float64, 1}
  @testset "$name" for (name, solve) in [
    ("R2", () -> R2(BroadcastQuartic(S))),
    ("R2 (M = 3)", () -> R2(BroadcastQuartic(S); M = 3)),
    ("TR", () -> TR(BroadcastQuartic(S))),
    ("fomo", () -> fomo(BroadcastQuartic(S))),
    ("fomo (M = 3)", () -> fomo(BroadcastQuartic(S); M = 3)),
    ("R2NLS + LSMR", () -> R2NLS(BroadcastQuarticNLS(S); subsolver = LSMRSubsolver)),
    ("R2NLS + LSQR", () -> R2NLS(BroadcastQuarticNLS(S); subsolver = LSQRSubsolver)),
    ("R2NLS + CGLS", () -> R2NLS(BroadcastQuarticNLS(S); subsolver = CGLSSubsolver)),
    (
      "R2NLS + LSMR (non_mono_size = 3)",
      () -> R2NLS(BroadcastQuarticNLS(S); subsolver = LSMRSubsolver, non_mono_size = 3),
    ),
  ]
    stats = solve()
    @test stats.status == :first_order
    @test stats.solution isa JLArray
    @test Array(stats.solution) ≈ ones(10) atol = 1e-4
  end
end
