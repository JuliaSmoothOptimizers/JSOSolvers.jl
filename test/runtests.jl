# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using NLPModelsTest

using Random

# this package
using JSOSolvers

@testset "Test small residual checks $solver" for solver in (:TrunkSolverNLS, :TronSolverNLS)
  nls = ADNLSModel(x -> [x[1] - 1; sin(x[2])], [-1.2; 1.0], 2)
  stats = GenericExecutionStats(nls)
  solver = eval(solver)(nls)
  SolverCore.solve!(solver, nls, stats, atol = 0.0, rtol = 0.0, Fatol = 1e-6, Frtol = 0.0)
  @test stats.status_reliable && stats.status == :small_residual
  @test stats.objective_reliable && isapprox(stats.objective, 0, atol = 1e-6)
end

@testset "Test iteration limit" begin
  @testset "$fun" for fun in (R2, R2N, fomo, lbfgs, tron, trunk)
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0])

    stats = eval(fun)(nlp, max_iter = 1)
    @test stats.status == :max_iter
  end

  @testset "$(fun)-NLS" for fun in (tron, trunk)
    f(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
    nlp = ADNLSModel(f, [-1.2; 1.0], 2)

    stats = eval(fun)(nlp, max_iter = 1)
    @test stats.status == :max_iter
  end
end

@testset "Test unbounded below" begin
  @testset "$fun" for fun in (R2, R2N, fomo, lbfgs, tron, trunk)
    T = Float64
    x0 = [T(0)]
    f(x) = -exp(x[1])
    nlp = ADNLPModel(f, x0)

    stats = eval(fun)(nlp)
    @test stats.status == :unbounded
    @test stats.objective < -one(T) / eps(T)
  end
end

include("restart.jl")
include("callback.jl")
include("consistency.jl")
include("test_solvers.jl")
if VERSION ≥ v"1.7"
  include("allocs.jl")

  @testset "Test warning for infeasible initial guess" begin
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + sin(x[2])^2, [-1.2; 1.0], zeros(2), ones(2))
    @test_warn "Warning: Initial guess is not within bounds." tron(nlp, verbose = 1)
    nls = ADNLSModel(x -> [x[1] - 1; sin(x[2])], [-1.2; 1.0], 2, zeros(2), ones(2))
    @test_warn "Warning: Initial guess is not within bounds." tron(nls, verbose = 1)
  end
end

include("objgrad-on-tron.jl")

@testset "Test max_radius in TRON" begin
  max_radius = 0.00314
  increase_factor = 5.0
  function cb(nlp, solver, stats)
    @test solver.tr.radius ≤ max_radius
  end

  nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
  stats = tron(nlp, max_radius = max_radius, increase_factor = increase_factor, callback = cb)

  nls = ADNLSModel(x -> [100 * (x[2] - x[1]^2); x[1] - 1], [-1.2; 1.0], 2)
  stats = tron(nls, max_radius = max_radius, increase_factor = increase_factor, callback = cb)
end

@testset "Preconditioner in Trunk" begin
  x0 = [-1.2; 1.0]
  nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, x0)
  function DiagPrecon(x)
    H = Matrix(hess(nlp, x))
    λmin = minimum(eigvals(H))
    Diagonal(H + (λmin + 1e-6) * I)
  end
  M = DiagPrecon(x0)
  function callback(nlp, solver, stats)
    M[:] = DiagPrecon(solver.x)
  end
  stats = trunk(nlp, callback = callback, M = M)
  @test stats.status == :first_order
end


# Test cases for solve_shifted_system!
@testset "solve_shifted_system! tests" begin
  # Seed the random number generator for reproducibility.
  Random.seed!(1234)

  # Set problem parameters.
  n    = 100      # dimension of the problem
  mem  = 10       # L-BFGS memory size
  scaling = true  # flag to enable scaling

  # Create an L-BFGS operator.
  B = LBFGSOperator(n; mem=mem, scaling=scaling)

  # Populate the L-BFGS operator with random {s, y} pairs.
  for i in 1:mem
      s = rand(n)
      y = rand(n)
      push!(B, s, y)
  end

  # Prepare the right-hand side and solution vector.
  b = rand(n)
  x = zeros(n)
  σ = 0.1   # a nonnegative shift

  # Solve the shifted system (in place).
  result = solve_shifted_system!(x, B, b, σ)

  # Test that the returned solution (stored in x) satisfies
  # (B + σ I)x ≈ b. We compute the relative residual.
  rel_residual = norm(B * x + σ * x - b) / norm(b)
  @test rel_residual < 1e-8

  # Additionally, test that the function updates x in place.
  @test result === x

  # Finally, verify that a negative shift throws an ArgumentError.
  x_temp = zeros(n)
  @test_throws ArgumentError solve_shifted_system!(x_temp, B, b, -0.1)
end

