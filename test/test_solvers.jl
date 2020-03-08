include("solvers/unconstrained.jl")
include("solvers/bound-constrained.jl")
include("solvers/unconstrained-nls.jl")

function test_solvers()
  @info "Testing NLP solvers"
  @info "  unconstrained solvers"
  for solver in [trunk, lbfgs, tron]
    @info "    $solver"
    test_unconstrained_solver(solver)
  end
  @info "  bound-constrained solvers"
  for solver in [tron]
    @info "    $solver"
    test_bound_constrained_solver(solver)
  end

  @info "Testing NLS solvers"
  @info "  unconstrained solvers"
  for (name,solver) in [
         ("trunk+cgls", (nls; kwargs...) -> trunk(nls, subsolver=:cgls; kwargs...)), # trunk with cgls due to multiprecision
         ("trunk full Hessian", (nls; kwargs...) -> trunk(nls, variant=:Newton; kwargs...))
       ]
    @info "    $name"
    test_unconstrained_nls_solver(solver)
  end
end

test_solvers()

@info "Specific solver tests"
include("solvers/trunkls.jl")
