function consistency()
  unlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, zeros(2))
  unls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], zeros(2), 2)

  @testset "Consistency" begin
    args = Pair{Symbol,Number}[:atol => 1e-6, :rtol => 1e-6, :max_eval => 1000, :max_time => 60.0]

    @testset "NLP" begin
      for Solver in [LBFGSSolver, TronSolver, TrunkSolver]
        with_logger(NullLogger()) do
          solver = Solver(unlp)
          output = solve!(solver, unlp; args...)
          @test output isa OptSolverOutput
          @test output.status == :first_order

          reset!(unlp)
          obj(unlp, unlp.meta.x0)
          obj(unlp, unlp.meta.x0)
          output = solve!(solver, unlp; max_eval=1)
          @test output.status == :max_eval

          slow_nlp = ADNLPModel(x -> begin sleep(0.1); unlp.f(x); end, unlp.meta.x0)
          solver = Solver(slow_nlp)
          output = solve!(solver, slow_nlp; max_time=0.0)
          @test output.status == :max_time
        end
      end
    end

    @testset "NLS" begin
      for Solver in [TronNLSSolver, TrunkNLSSolver]
        with_logger(NullLogger()) do
          solver = Solver(unls)
          output = solve!(solver, unls; args...)
          @test output isa OptSolverOutput
          @test output.status == :first_order

          reset!(unls)
          obj(unls, unls.meta.x0)
          obj(unls, unls.meta.x0)
          output = solve!(solver, unls; max_eval=1)
          @test output.status == :max_eval

          slow_nls = ADNLSModel(x -> begin sleep(0.1); unls.F(x); end, unls.meta.x0, nls_meta(unls).nequ)
          solver = Solver(slow_nls)
          output = solve!(solver, slow_nls; max_time=0.0)
          @test output.status == :max_time
        end
      end
    end

  end
end

consistency()
