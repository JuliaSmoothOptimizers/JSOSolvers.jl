mutable struct SimpleModel{T, S} <: AbstractNLPModel{T, S}
  meta :: NLPModelMeta{T, S}
  counters :: Counters
end

SimpleModel(n :: Int) = SimpleModel(NLPModelMeta(n, x0=ones(n)), Counters())

function NLPModels.obj(nlp::SimpleModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  sum(xi ^ 4 for xi in x) / 12
end

function NLPModels.grad!(nlp::SimpleModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  @. g = x ^ 3 / 3
  g
end

function NLPModels.objgrad!(nlp::SimpleModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  @. g = x ^ 3 / 3
  return sum(xi ^4 for xi in x) / 12, g
end

function NLPModels.hprod!(nlp::SimpleModel, x::AbstractVector{T}, v::AbstractVector, Hv::AbstractVector; obj_weight::T = one(T)) where T
  increment!(nlp, :neval_hprod)
  @. Hv = obj_weight * x ^ 2 * v
  Hv
end

function NLPModels.hess(nlp::SimpleModel, x::AbstractVector{T}; obj_weight::T = one(T)) where T
  increment!(nlp, :neval_hprod)
  return obj_weight .* diagm(0 => x .^ 2)
end