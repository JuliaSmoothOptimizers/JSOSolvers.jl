# stdlib
using Printf, LinearAlgebra, Logging, SparseArrays, Test
# additional packages
using ADNLPModels, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, Krylov
using NLPModelsTest, SolverParameters

# this package
using JSOSolvers

@testset "Test parameterset" begin
  @testset "Test unconstrained parameters $paramset" for (paramset, fun) in (
    (LBFGSParameterSet, lbfgs),
    (TRONParameterSet, tron),
    (TRUNKParameterSet, trunk),
    (FOMOParameterSet, fomo),
    (R2NParameterSet, R2N),
  )
    nlp = BROWNDEN()
    params = eval(paramset)(nlp)
    args = Dict(
      sym => SolverParameters.value(getfield(params, sym)) for sym in fieldnames(typeof(params))
    )
    stats = fun(nlp; args...)
    @test stats.status == :first_order
  end

  @testset "Test unconstrained NLS parameters $paramset" for (paramset, fun) in (
    (TRONLSParameterSet, tron),
    (TRUNKLSParameterSet, trunk),
    (R2NLSParameterSet, R2NLS),
  )
    nls = MGH01()
    params = eval(paramset)(nls)
    args = Dict(
      sym => SolverParameters.value(getfield(params, sym)) for sym in fieldnames(typeof(params))
    )
    stats = fun(nls; args...)
    @test stats.status == :first_order
  end
end

@testset "Test small residual checks $solver" for solver in (:TrunkSolverNLS, :TronSolverNLS, :R2NLSSolver)
  nls = ADNLSModel(x -> [x[1] - 1; sin(x[2])], [-1.2; 1.0], 2)
  stats = GenericExecutionStats(nls)
  solver = eval(solver)(nls)
  SolverCore.solve!(solver, nls, stats, atol = 0.0, rtol = 0.0, Fatol = 1e-6, Frtol = 0.0)
  @test stats.status_reliable && stats.status == :small_residual
  @test stats.objective_reliable && isapprox(stats.objective, 0, atol = 1e-6)
end

@testset "Test R2N direct subsolver guard" begin
  nls = ADNLSModel(x -> [x[1] - 1; 2 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  @test !is_unsupported(QRMumpsSubsolver(nls))
  @test is_unsupported(QRMumpsSubsolver(nls; min_matrix_size = 0))
end

@testset "Test R2N regularization lower bounds" begin
  σmin = 10.0
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  r2n_callback_called = Ref(false)
  function r2n_sigma_callback(nlp, solver, stats)
    r2n_callback_called[] = true
    @test solver.σ >= σmin
  end
  R2N(nlp; σmin = σmin, callback = r2n_sigma_callback, max_iter = 3)
  @test r2n_callback_called[]

  nls = ADNLSModel(x -> [x[1] - 1; 2 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  r2nls_callback_called = Ref(false)
  function r2nls_sigma_callback(nls, solver, stats)
    r2nls_callback_called[] = true
    @test solver.σ >= σmin
  end
  R2NLS(
    nls;
    σmin = σmin,
    subsolver = LSMRSubsolver,
    callback = r2nls_sigma_callback,
    max_iter = 3,
  )
  @test r2nls_callback_called[]
end

@testset "Test iteration limit" begin
  @testset "$fun" for fun in (R2, R2N, fomo, lbfgs, tron, trunk)
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0])

    stats = eval(fun)(nlp, max_iter = 1)
    @test stats.status == :max_iter
  end

  @testset "$(fun)-NLS" for fun in (R2NLS, tron, trunk)
    f(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
    nlp = ADNLSModel(f, [-1.2; 1.0], 2)

    stats = eval(fun)(nlp, max_iter = 1)
    @test stats.status == :max_iter
  end
end

@testset "Test unbounded below" begin
  @testset "$name" for (name, solver) in [
    ("trunk", trunk),
    ("lbfgs", lbfgs),
    ("tron", tron),
    ("R2", R2),
    ("R2N", R2N),
    (
      "R2N_ShiftedLBFGS",
      (nlp; kwargs...) ->
        R2N(LBFGSModel(nlp), subsolver = ShiftedLBFGSSolver; kwargs...),
    ),
    ("fomo", fomo),
  ]
    T = Float64
    x0 = [T(0)]
    f(x) = -exp(x[1])
    nlp = ADNLPModel(f, x0)

    stats = solver(nlp)
    @test stats.status == :unbounded
    @test stats.objective < -one(T) / eps(T)
  end
end

include("test_hsl_subsolver.jl")
include("restart.jl")
include("callback.jl")
include("consistency.jl")
include("test_solvers.jl")
include("incompatible.jl")

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