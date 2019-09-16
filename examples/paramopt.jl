using NLPModels, JSOSolvers, BenchmarkProfiles, Plots, SolverBenchmark, SolverTools
pyplot()

#= This is an example of parameter optimization.
=#
function paramopt()
  optsolver = SolverLBFGS()

  problems = Any[]
  for p = 1:100
    n = rand(2:100)
    Λ = 1e-4 .+ rand(n) ./ rand(n)
    push!(problems, ADNLPModel(x -> dot(x, Λ .* x), ones(n)))
  end

  optsolver(problems[1])

  tune!(optsolver, problems, bbmax_f=100,
        solver_args = Dict(:max_eval=>1000, :max_time=>3.0))

  solver = SolverLBFGS()
  solvers = Dict(:LBFGS => solver, :LBFGS_tuned => optsolver)
  stats = bmark_solvers(solvers, problems, max_eval=1000, max_time=3.0)

  #= For debugging
  for (k, df) in stats
    println("Solver $k")
    markdown_table(stdout, df)
  end
  =#

  cost(df) = (df.status .!= :first_order) * Inf + df.neval_obj
  performance_profile(stats, cost)
  png("paramopt-lbfgs")
end

paramopt()
