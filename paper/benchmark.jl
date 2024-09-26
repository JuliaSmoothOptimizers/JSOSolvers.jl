using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using NLPModels, NLPModelsIpopt, JSOSolvers
using SolverBenchmark

###############################################################################
# Benchmark selection
using CUTEst
using ADNLPModels, NLPModelsJuMP, OptimizationProblems
df = OptimizationProblems.meta
unconstrained_problems = df[(df.contype .== :unconstrained) .& (.!df.has_bounds), :]

if length(ARGS) > 0
  run_cutest = run_jump = run_ad = run_ad_nls = false
  if ARGS[1] == "cutest"
    run_cutest = true
  elseif ARGS[1] == "jump"
    run_jump = true
  elseif ARGS[1] == "ad"
    run_ad = true
  elseif ARGS[1] == "adnls"
    run_ad_nls = true
  else
    @info "Unknown input"
  end
else
  run_cutest = false
  run_jump = false
  run_ad = true
  run_ad_nls = false
end
# Add scalable benchmark? (meta.variable_nvar .== true)
T = Float64 # may be changed for ADNLPModels benchmark only

max_time = 10.0 # 20 minutes
tol = 1e-5 # relative tolerance
###############################################################################

(problems, bench) = if run_cutest
  ((CUTEstModel(p) for p in readlines("list_problems.dat")), "cutest")
elseif run_jump
  (
    (
      MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Meta.parse(problem))(), name = problem)
      for problem ∈ unconstrained_problems[!, :name]
    ),
    "juop",
  )
elseif run_ad
  (
    (
      OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(type = T) for
      problem ∈ unconstrained_problems[!, :name]
    ),
    "adop",
  )
else # run_ad_nls
  (
    (
      eval(Meta.parse(problem))(use_nls = true, type = T) for
      problem ∈ unconstrained_problems[(df.objtype .== :least_squares), :name]
    ),
    "adopnls",
  )
end

solvers = Dict(
  :ipopt =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      dual_inf_tol = Inf,
      constr_viol_tol = Inf,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      max_cpu_time = max_time,
      tol = tol,
    ),
  :lbfgs => model -> lbfgs(model, atol = 0.0, rtol = tol),
  :tron => model -> trunk(model, atol = 0.0, rtol = tol),
  :trunk => model -> trunk(model, atol = 0.0, rtol = tol),
)

stats = bmark_solvers(
  solvers,
  problems,
  #skipif=prob -> (get_nvar(prob) < 3) # useful for debugging
)

using JLD2, Dates
@save "$(today())_ipopt_lbfgs_trunk_tron_$(bench)_$(T)_$(string(length(problems))).jld2" stats max_time tol
