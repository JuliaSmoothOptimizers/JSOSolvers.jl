# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using NLPModelsTest

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
  @testset "$fun" for fun in (R2, lbfgs, tron, trunk)
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
