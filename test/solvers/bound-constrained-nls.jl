function test_bound_constrained_nls_solver(solver)
  @testset "Simple test" begin
    @testset "Small quadratic" begin
      x0 = [1.0; 2.0]
      F(x) = x
      l = [0.5; 0.25]
      u = [1.2; 1.5]

      nls = ADNLSModel(F, x0, 2, l, u)

      stats = with_logger(NullLogger()) do
        solver(nls)
      end
      @test stats.solution ≈ l
      @test stats.objective ≈ norm(F(l))^2 / 2
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Rosenbrock inactive bounds" begin
      l = [0.5; 0.25]
      u = [1.2; 1.5]
      x0 = (l + u) / 2
      F(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]

      nls = ADNLSModel(F, x0, 2, l, u)

      stats = with_logger(NullLogger()) do
        solver(nls)
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
      F(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]

      nls = ADNLSModel(F, x0, 2, l, u)

      sol = [0.9; 0.81]

      stats = with_logger(NullLogger()) do
        solver(nls)
      end
      @test isapprox(stats.solution, sol, atol=1e-3)
      @test isapprox(stats.objective, norm(F(sol))^2 / 2, atol=1e-5)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end
  end

  @testset "Fixed variables" begin
    @testset "One fixed" begin
      x0 = ones(3)
      l = [1.0; 0.0; 0.0]
      u = [1.0; 2.0; 2.0]
      F(x) = x .- 3
      nls = ADNLSModel(F, x0, 3, l, u)
      stats = with_logger(NullLogger()) do
        solver(nls)
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
      F(x) = x.^2
      nls = ADNLSModel(F, x0, n, l, u)
      stats = with_logger(NullLogger()) do
        solver(nls)
      end
      @test stats.status == :first_order
      @test stats.dual_feas == 0.0
      @test stats.solution == l
    end
  end

  @testset "Extended Rosenbrock" begin
    n = 30
    F(x) = [[10 * (x[i+1] - x[i]^2) for i = 1:n-1];
            [x[i] - 1 for i = 1:n-1]]
    nls = ADNLSModel(F, collect((1:n) ./ (n+1)), 2n-2, zeros(n), 0.3*ones(n))

    stats = with_logger(NullLogger()) do
      solver(nls, max_time=30.0)
    end
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Nonlinear regression of COVID19 cases in São Paulo, Brazil" begin
    y = [136, 152, 164, 240, 286, 345, 459, 631, 745, 810, 862, 1052, 1223, 1406, 1451, 1517]
    n = length(y)
    x = collect(1:n)
    h(β, x) = β[3] * exp(β[1] + β[2] * x) / (1 + exp(β[1] + β[2] * x))
    r(β) = [y[i] - h(β, x[i]) for i = 1:n]
    lvar = [-Inf; -Inf; 2y[end]]
    uvar = [ 0.0;  Inf;     Inf]
    nls = ADNLSModel(r, [-1.0; 1.0; 3y[end]], n, lvar, uvar)

    stats = with_logger(NullLogger()) do
      solver(nls, max_time=30.0)
    end
    @test stats.dual_feas < 1e-6 * norm(grad(nls, nls.meta.x0))
    @test stats.status == :first_order
  end

  @testset "Multiprecision" begin
    F(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]
    for T in (Float16, Float32, Float64, BigFloat)
      nls = ADNLSModel(F, T[-1.2; 1.0], 2, zeros(T, 2),  ones(T, 2) / 2)
      ϵ = eps(T)^T(1/4)

      g0 = grad(nls, nls.meta.x0)
      ng0 = norm(g0)

      stats = with_logger(NullLogger()) do
        solver(nls, max_eval=-1, atol=ϵ, rtol=ϵ)
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
