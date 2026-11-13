@testset "lbfgs mem = :auto" begin
  nlp = BROWNDEN()

  @testset "mem is recorded in the stats" begin
    stats = lbfgs(nlp; mem = 7)
    @test stats.status == :first_order
    @test stats.solver_specific[:mem] == 7
    @test lbfgs(nlp).solver_specific[:mem] == 5
  end

  @testset "invalid symbol" begin
    @test_throws ArgumentError lbfgs(nlp; mem = :foo)
    @test_throws ArgumentError LBFGSSolver(nlp; mem = :foo)
  end

  if isempty(methods(JSOSolvers.lbfgs_auto_mem))
    @testset ":auto without LBFGSMemTuner falls back to the default" begin
      stats = @test_logs (:warn, r"LBFGSMemTuner") lbfgs(nlp; mem = :auto)
      @test stats.status == :first_order
      @test stats.solver_specific[:mem] == 5
    end
  else
    @testset ":auto with LBFGSMemTuner" begin
      stats = lbfgs(nlp; mem = :auto)
      @test stats.status == :first_order
      @test 1 <= stats.solver_specific[:mem] <= 100
      solver = LBFGSSolver(nlp; mem = :auto)
      @test 1 <= SolverParameters.value(solver.params.mem) <= 100
      @test solve!(solver, nlp).status == :first_order
    end
  end
end
