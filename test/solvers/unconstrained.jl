function test_unconstrained_solver(solver)
  @testset "Simple tests" begin
    @testset "f(x) = x₁² + x₂²" begin
      nlp = ADNLPModel(x->dot(x, x), ones(2))

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test isapprox(stats.solution, zeros(2), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-16)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Diagonal quadratic" begin
      n = 100
      D = [0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n]
      nlp = ADNLPModel(x->dot(x, D .* x), ones(n))

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test isapprox(stats.solution, zeros(n), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Tridiagonal quadratic" begin
      n = 100
      A = spdiagm(0 => 2 * ones(n), -1 => -ones(n-1), -1 => -ones(n-1))
      nlp = ADNLPModel(x->dot(x, A * x), ones(n))

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test isapprox(stats.solution, zeros(n), atol=1e-4)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Rosenbrock" begin
      nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])

      stats = with_logger(NullLogger()) do
        solver(nlp, max_eval=-1)
      end
      @test isapprox(stats.solution, ones(2), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end
  end

  @testset "Extended Rosenbrock" begin
    n = 30
    nlp = ADNLPModel(x->sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:n-1), (1:n) ./ (n + 1))

    ng0 = norm(grad(nlp, nlp.meta.x0))
    stats = with_logger(NullLogger()) do
      solver(nlp, max_time=30.0, max_eval=-1)
    end
    @test isapprox(stats.solution, ones(n), atol=1e-6)
    @test isapprox(stats.objective, 0.0, atol=1e-6)
    @test stats.dual_feas < √eps() * (ng0 + 1)
    @test stats.status == :first_order
  end

  @testset "Multiprecision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x -> (x[1] - 1)^2 + (x[2] - x[1]^2)^2, T[-1.2; 1.0])
      ϵ = eps(T)^T(1/4)

      ng0 = norm(grad(nlp, nlp.meta.x0))

      stats = with_logger(NullLogger()) do
        solver(nlp, max_eval=-1, atol=ϵ, rtol=ϵ)
      end
      @test eltype(stats.solution) == T
      @test stats.objective isa T
      @test stats.dual_feas isa T
      @test stats.primal_feas isa T
      @test isapprox(stats.solution, ones(T, 2), atol=ϵ * ng0 * 10)
      @test isapprox(stats.objective, zero(T), atol=ϵ * ng0)
      @test stats.dual_feas < ϵ * ng0 + ϵ
    end
  end
end
