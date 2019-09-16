# TODO: Store `solver`, `problems` here instead on inside `tune`
mutable struct TuningProblem <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  f :: Function
  c :: Function
  # TODO: Constraints for solver parameters
  # TODO: Constraints for tuning problem
end

function TuningProblem(f, x0, lvar, uvar, c, lcon, ucon)
  meta = NLPModelMeta(length(x0), x0=x0, lvar=lvar, uvar=uvar, ncon=length(lcon), lcon=lcon, ucon=ucon, nnzj=0, nnzh=0)
  return TuningProblem(meta, Counters(), f, c)
end

function NLPModels.obj(nlp::TuningProblem, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function NLPModels.cons(nlp::TuningProblem, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_cons)
  return nlp.c(x)
end
