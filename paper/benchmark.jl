using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using NLPModels, NLPModelsIpopt, JSOSolvers
using SolverBenchmark

# Benchmark selection
using CUTEst

cutest_problems = readlines("list_problems.dat")
problems = (CUTEstModel(p) for p in cutest_problems)

max_time = 1200.0 # 20 minutes
T = Float64
tol = sqrt(eps(T)) # relative tolerance

solvers = Dict(
  :ipopt =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      #dual_inf_tol = Inf,
      #constr_viol_tol = Inf,
      #compl_inf_tol = Inf,
      #acceptable_iter = 0,
      max_cpu_time = max_time,
      tol = tol,
    ),
  :lbfgs => model -> lbfgs(model, atol = zero(T), rtol = tol, max_time = max_time),
  :tron => model -> tron(model, atol = zero(T), rtol = tol, max_time = max_time),
  :trunk => model -> trunk(model, atol = zero(T), rtol = tol, max_time = max_time),
)
name_solvers = "ipopt_lbfgs_trunk_tron"

stats = bmark_solvers(
  solvers,
  problems,
  #skipif=prob -> (prob.meta.name == "penalty2"), #(get_nvar(prob) < 3) # useful for debugging
)

using JLD2, Dates
@save "$(today())_$(name_solvers)_cutest_$(string(length(problems))).jld2" stats max_time tol
