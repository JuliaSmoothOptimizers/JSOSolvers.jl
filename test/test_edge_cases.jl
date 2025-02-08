# 1. Test error for constrained problems
@testset "Constrained Problems Error" begin
  f(x) = (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2
  x0 = [-1.2; 1.0]
  lvar = [-Inf; 0.1]
  uvar = [0.5; 0.5]
  c(x) = [x[1] + x[2] - 2; x[1]^2 + x[2]^2]
  lcon = [0.0; -Inf]
  ucon = [Inf; 1.0]
  nlp_constrained = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

  solver = R2NSolver(nlp_constrained)
  stats = GenericExecutionStats(nlp_constrained)

  @test_throws ErrorException begin
    SolverCore.solve!(solver, nlp_constrained, stats)
  end
end

# 2. Test error when non_mono_size < 1
@testset "non_mono_size < 1 Error" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  @test_throws ErrorException begin
    R2NSolver(nlp; non_mono_size = 0)
  end
  @test_throws ErrorException begin
    R2N(nlp; non_mono_size = 0)
  end

  @test_throws ErrorException begin
    R2NSolver(nlp; non_mono_size = -1)
  end
  @test_throws ErrorException begin
    R2N(nlp; non_mono_size = -1)
  end
end

# 3. Test error when subsolver_type is ShiftedLBFGSSolver but nlp is not of type LBFGSModel
@testset "ShiftedLBFGSSolver with wrong nlp type Error" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  @test_throws ErrorException begin
    R2NSolver(nlp; subsolver_type = ShiftedLBFGSSolver)
  end
  @test_throws ErrorException begin
    R2N(nlp; subsolver_type = ShiftedLBFGSSolver)
  end
end

# 4. Test error when subsolver_type is not a subtype of R2N_allowed_subsolvers
@testset "Invalid subsolver type Error" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  @test_throws ErrorException begin
    R2NSolver(nlp; subsolver_type = CgSolver)
  end
  @test_throws ErrorException begin
    R2N(nlp; subsolver_type = CgSolver)
  end
end
