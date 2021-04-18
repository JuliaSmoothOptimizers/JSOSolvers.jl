using SolverTest

function tests()
  @info "Testing NLP solvers"
  @info "  unconstrained solvers"
  for Solver in [LBFGSSolver, TronSolver, TrunkSolver]
    @info "    $Solver"
    unconstrained_nlp(Solver)
    multiprecision_nlp(Solver, :unc)
  end
  @info "  bound-constrained solvers"
  for Solver in [TronSolver]
    @info "    $Solver"
    bound_constrained_nlp(Solver)
    multiprecision_nlp(Solver, :unc)
    multiprecision_nlp(Solver, :bnd)
  end

  @info "Testing NLS solvers"
  @info "  unconstrained solvers"
  for Solver in [
    TronNLSSolver,
    TronSolver,
    TrunkNLSSolver,
    TrunkSolver,
  ]
    @info "    $Solver"
    unconstrained_nls(Solver)
    multiprecision_nls(Solver, :unc)
  end
  @info "  bound-constrained solvers"
  for Solver in [
    TronNLSSolver,
    TronSolver
  ]
    @info "    $Solver"
    bound_constrained_nls(Solver)
    multiprecision_nls(Solver, :unc)
    multiprecision_nls(Solver, :bnd)
  end
end

tests()

@info "Specific solver tests"
include("solvers/tronls.jl")
include("solvers/trunkls.jl")
include("incompatible.jl")
