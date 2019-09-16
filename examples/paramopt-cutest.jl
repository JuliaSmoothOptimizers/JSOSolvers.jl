using CUTEst, JSOSolvers, BenchmarkProfiles, Plots, SolverBenchmark, SolverTools
pyplot()

#= This is an example of parameter optimization.
=#
function paramopt()
  optsolver = SolverLBFGS()
  ps = CUTEst.select(max_var=10, max_con=0, only_free_var=true)
  problems = (CUTEstModel(p) for p in ps)

  tune!(optsolver, problems, bbmax_f=100,
        solver_args = Dict(:max_eval=>1000, :max_time=>3.0))

  solver = SolverLBFGS()
  solvers = Dict(:LBFGS => solver, :LBFGS_tuned => optsolver)
  stats = bmark_solvers(solvers, problems, max_eval=1000, max_time=3.0)

  cost(df) = (df.status .!= :first_order) * Inf + df.neval_obj
  performance_profile(stats, cost)
  png("paramopt-lbfgs")
end

paramopt()
