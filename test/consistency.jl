function consistency()
  unlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, zeros(2))
  unls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], zeros(2), 2)

  @testset "Consistency" begin
    args = Pair{Symbol, Number}[:atol => 1e-6, :rtol => 1e-6, :max_eval => 1000, :max_time => 60.0]

    @testset "NLP" begin
      for mtd in [trunk, lbfgs, tron]
        with_logger(NullLogger()) do
          stats = mtd(unlp; args...)
          @test stats isa GenericExecutionStats
          @test stats.status == :first_order
          reset!(unlp)
          stats = mtd(unlp; max_eval = 1)
          @test stats.status == :max_eval
          slow_nlp = ADNLPModel(x -> begin
            sleep(0.1)
            unlp.f(x)
          end, unlp.meta.x0)
          stats = mtd(slow_nlp; max_time = 0.0)
          @test stats.status == :max_time
        end
      end
    end

    @testset "NLS" begin
      for mtd in [trunk]
        with_logger(NullLogger()) do
          stats = mtd(unls; args...)
          @test stats isa GenericExecutionStats
          @test stats.status == :first_order
          reset!(unls)
          stats = mtd(unls; max_eval = 1)
          @test stats.status == :max_eval
          slow_nls = ADNLSModel(x -> begin
            sleep(0.1)
            unls.F(x)
          end, unls.meta.x0, nls_meta(unls).nequ)
          stats = mtd(slow_nls; max_time = 0.0)
          @test stats.status == :max_time
        end
      end
    end
  end
end

consistency()
