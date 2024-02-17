using ADNLPModels, JSOSolvers, LinearAlgebra, Logging #, Plots
@testset "Test callback" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]
  function cb(nlp, solver, stats)
    x = solver.x
    push!(X, x[1])
    push!(Y, x[2])
    if stats.iter == 8
      stats.status = :user
    end
  end
  stats = with_logger(NullLogger()) do
    R2(nlp, callback = cb)
  end
  @test stats.iter == 8

  stats = with_logger(NullLogger()) do
    lbfgs(nlp, callback = cb)
  end
  @test stats.iter == 8

  stats = with_logger(NullLogger()) do
    trunk(nlp, callback = cb)
  end
  @test stats.iter == 8

  stats = with_logger(NullLogger()) do
    tron(nlp, callback = cb)
  end
  @test stats.iter == 8

  stats = with_logger(NullLogger()) do
    fomo(nlp, callback = cb)
  end
  @test stats.iter == 8
end

@testset "Test callback for NLS" begin
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nls = ADNLSModel(F, [-1.2; 1.0], 2)
  function cb(nlp, solver, stats)
    if stats.iter == 8
      stats.status = :user
    end
  end

  stats = with_logger(NullLogger()) do
    trunk(nls, callback = cb)
  end
  @test stats.iter == 8

  stats = with_logger(NullLogger()) do
    tron(nls, callback = cb)
  end
  @test stats.iter == 8
end

@testset "Testing Solver Values" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  function cb(nlp, solver, stats)
    if stats.iter == 4
      @test solver.α > 0.0
      stats.status = :user
    end
  end
  stats = with_logger(NullLogger()) do
    R2(nlp, callback = cb)
  end
end
