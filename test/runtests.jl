# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using NLPModelsTest

# this package
using JSOSolvers

include("restart.jl")
include("callback.jl")
include("consistency.jl")
include("test_solvers.jl")
if VERSION ≥ v"1.6"
  include("allocs.jl")
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
