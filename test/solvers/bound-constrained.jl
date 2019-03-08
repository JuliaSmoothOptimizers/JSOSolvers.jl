function test_bound_constrained_solver(solver)
  @testset "Simple test" begin
    @testset "Small quadratic" begin
      x0 = [1.0; 2.0]
      f(x) = dot(x,x)/2
      l = [0.5; 0.25]
      u = [1.2; 1.5]

      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test stats.solution ≈ l
      @test stats.objective == f(l)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Rosenbrock inactive bounds" begin
      l = [0.5; 0.25]
      u = [1.2; 1.5]
      x0 = (l + u) / 2
      f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2

      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test isapprox(stats.solution, [1.0; 1.0], atol=1e-3)
      @test isapprox(stats.objective, 0.0, atol=1e-5)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Rosenbrock active bounds" begin
      l = [0.5; 0.25]
      u = [0.9; 1.5]
      x0 = (l + u) / 2
      f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2

      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

      sol = [0.9; 0.81]

      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test isapprox(stats.solution, sol, atol=1e-3)
      @test isapprox(stats.objective, f(sol), atol=1e-5)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end
  end

  @testset "Fixed variables" begin
    @testset "One fixed" begin
      x0 = ones(3)
      l = [1.0; 0.0; 0.0]
      u = [1.0; 2.0; 2.0]
      f(x) = 0.5*dot(x .- 3, x .- 3)
      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test stats.status == :first_order
      @test stats.dual_feas < 1e-6
      @test isapprox(stats.solution, [1.0; 2.0; 2.0], atol=1e-3)
      @test stats.solution[1] == 1.0
    end

    @testset "All fixed" begin
      n = 100
      x0 = zeros(n)
      l = 0.9*ones(n)
      u = copy(l)
      f(x) = sum(x.^4)
      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
      stats = with_logger(NullLogger()) do
        solver(nlp)
      end
      @test stats.status == :first_order
      @test stats.dual_feas == 0.0
      @test stats.solution == l
    end
  end

  @testset "Extended Rosenbrock" begin
    n = 30
    nlp = ADNLPModel(x->sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:n-1), (1:n) ./ (n+1),
                     lvar=zeros(n), uvar=0.3*ones(n))

    stats = with_logger(NullLogger()) do
      solver(nlp, max_time=30.0)
    end
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Multiprecision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x -> (x[1] - 1)^2 + (x[2] - x[1]^2)^2, T[-1.2; 1.0],
                       lvar=zeros(T, 2), uvar= ones(T, 2) / 2)
      ϵ = eps(T)^T(1/4)

      g0 = grad(nlp, nlp.meta.x0)
      ng0 = norm(g0)

      stats = with_logger(NullLogger()) do
        solver(nlp, max_eval=-1, atol=ϵ, rtol=ϵ)
      end
      @test eltype(stats.solution) == T
      @test stats.objective isa T
      @test stats.dual_feas isa T
      @test stats.primal_feas isa T
      @test isapprox(stats.solution, [0.5; 0.25], atol=ϵ * ng0 * 10)
      @test stats.dual_feas < ϵ * ng0 + ϵ
    end
  end
end
