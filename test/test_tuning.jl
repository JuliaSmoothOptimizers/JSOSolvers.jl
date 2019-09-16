function test_tuning()
  problems = Any[]
  for p = 1:10
    n = rand(2:10)
    Λ = 1e-4 .+ rand(n) ./ rand(n)
    push!(problems, ADNLPModel(x -> dot(x, Λ .* x), ones(n)))
  end

  @testset "Testing tuning of solvers" begin
    for solver in [SolverLBFGS, SolverTrunk, SolverTRON]
      @testset "Testing tuning of $solver" begin
        optsolver = solver()
        tune!(optsolver, problems, max_eval=10)
        simple_solver = solver()
        solvers = Dict(:tuned => optsolver, :simple => simple_solver)
        stats = bmark_solvers(solvers, problems)
        cost(df) = dot(1.0 .+ (df.status .!= :first_order) * 9, df.neval_obj)
        @test cost(stats[:simple]) ≥ cost(stats[:tuned])
      end
    end
  end

  @testset "Testing tuning on subset of parameters" begin
    optsolver = SolverLBFGS()
    tune!(optsolver, problems, parameters = [:ls_acceptance], max_eval = 10)
  end
end

test_tuning()
