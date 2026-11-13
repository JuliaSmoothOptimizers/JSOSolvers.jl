module JSOSolversLBFGSMemTunerExt

using JSOSolvers, LBFGSMemTuner

# `lbfgs(nlp; mem = :auto)`: recommended memory from the two-stage model of LBFGSMemTuner.jl.
# Falls back to the JSOSolvers default (with a warning) if the problem features cannot be extracted.
JSOSolvers.lbfgs_auto_mem(nlp) =
  LBFGSMemTuner.auto_mem(nlp; fallback = Int(get(JSOSolvers.LBFGS_mem, nlp)))

end
