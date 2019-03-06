function test_unconstrained_nls_solver(solver)
  @testset "Simple tests" begin
    @testset "f(x) = x₁² + x₂²" begin
      nls = ADNLSModel(x->x, ones(2), 2)

      stats = with_logger(NullLogger()) do
        solver(nls)
      end
      @test isapprox(stats.solution, zeros(2), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-16)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Diagonal quadratic" begin
      n = 100
      D = [0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n]
      nls = ADNLSModel(x->sqrt.(D) .* x, ones(n), n)

      stats = with_logger(NullLogger()) do
        solver(nls)
      end
      @test isapprox(stats.solution, zeros(n), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Rosenbrock" begin
      nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)

      stats = with_logger(NullLogger()) do
        solver(nls, max_eval=-1)
      end
      @test isapprox(stats.solution, ones(2), atol=1e-6)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Underdeterminded" begin
      nls = ADNLSModel(x -> [x[2] - x[1]^2], [-1.2; 1.0], 1)

      stats = with_logger(NullLogger()) do
        solver(nls, max_eval=-1)
      end
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end

    @testset "Overdeterminded" begin
      nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2); x[1] * x[2] - 1], [-1.2; 1.0], 3)

      stats = with_logger(NullLogger()) do
        solver(nls, max_eval=-1)
      end
      @test isapprox(stats.solution, ones(2), atol=1e-4)
      @test isapprox(stats.objective, 0.0, atol=1e-6)
      @test stats.dual_feas < 1e-6
      @test stats.status == :first_order
    end
  end

  @testset "Extended Rosenbrock" begin
    n = 30
    nls = ADNLSModel(x->[[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]], (1:n) ./ (n + 1), 2n - 2)

    stats = with_logger(NullLogger()) do
      solver(nls, max_time=600.0, max_eval=-1)
    end
    @test isapprox(stats.solution, ones(n), atol=1e-6)
    @test isapprox(stats.objective, 0.0, atol=1e-6)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Multiprecision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLSModel(x -> [x[1] - 1; x[2] - x[1]^2], T[-1.2; 1.0], 2)
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
      @test isapprox(stats.solution, ones(T, 2), atol=ϵ * ng0 * 10)
      @test isapprox(stats.objective, zero(T), atol=ϵ * ng0)
      @test stats.dual_feas < ϵ * ng0 + ϵ
    end
  end

  @static if Sys.isunix()
    @testset "CUTEst" begin
      problems = CUTEst.select(max_var=2, objtype=:none, only_free_var=true, only_equ_con=true)
      stline = statshead([:objective, :dual_feas, :elapsed_time, :iter, :neval_obj,
                          :neval_grad, :neval_hprod, :status])
      @info @sprintf("%8s  %5s  %4s  %s\n", "Problem", "n", "type", stline)
      for p in problems
        nls = FeasibilityResidual(CUTEstModel(p))
        stats = with_logger(NullLogger()) do
          solver(nls, max_time=3.0)
        end
        finalize(nls)

        ctype = "unc"
        stline = statsline(stats, [:objective, :dual_feas, :elapsed_time, :iter, :neval_obj,
                                   :neval_grad, :neval_hprod, :status])
        @info @sprintf("%8s  %5d  %4s  %s\n", p, nls.meta.nvar, ctype, stline)
      end
    end
  end
end
