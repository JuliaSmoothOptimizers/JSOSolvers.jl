@testset "Test restart with a different initial guess: $fun" for (fun, s) in (
  (:R2, :FoSolver),
  (:fomo, :FomoSolver),
  (:lbfgs, :LBFGSSolver),
  (:tron, :TronSolver),
  (:trunk, :TrunkSolver),
)
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  stats = GenericExecutionStats(nlp)
  solver = eval(s)(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Test restart NLS with a different initial guess: $fun" for (fun, s) in (
  (:tron, :TronSolverNLS),
  (:trunk, :TrunkSolverNLS),
)
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  solver = eval(s)(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Test restart with a different problem: $fun" for (fun, s) in (
  (:R2, :FoSolver),
  (:fomo, :FomoSolver),
  (:lbfgs, :LBFGSSolver),
  (:tron, :TronSolver),
  (:trunk, :TrunkSolver),
)
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  stats = GenericExecutionStats(nlp)
  solver = eval(s)(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  f2(x) = (x[1])^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f2, [-1.2; 1.0])
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end

@testset "Test restart NLS with a different problem: $fun" for (fun, s) in (
  (:tron, :TronSolverNLS),
  (:trunk, :TrunkSolverNLS),
)
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  solver = eval(s)(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  F2(x) = [x[1]; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F2, [-1.2; 1.0], 2)
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end
