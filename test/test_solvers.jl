using SolverTest

function tests()
    @info "Testing NLP solvers"
    @info "  unconstrained solvers"
    for solver in [trunk, lbfgs, tron]
        @info "    $solver"
        unconstrained_nlp(solver)
        multiprecision_nlp(solver, :unc)
    end
    @info "  bound-constrained solvers"
    for solver in [tron]
        @info "    $solver"
        bound_constrained_nlp(solver)
        multiprecision_nlp(solver, :unc)
        multiprecision_nlp(solver, :bnd)
    end

    @info "Testing NLS solvers"
    @info "  unconstrained solvers"
    for (name, solver) in [
         ("trunk+cgls", (nls; kwargs...) -> trunk(nls, subsolver=:cgls; kwargs...)), # trunk with cgls due to multiprecision
         ("trunk full Hessian", (nls; kwargs...) -> trunk(nls, variant=:Newton; kwargs...)),
         ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver=:cgls; kwargs...)),
         ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant=:Newton; kwargs...))
       ]
        @info "    $name"
        unconstrained_nls(solver)
        multiprecision_nls(solver, :unc)
    end
    @info "  bound-constrained solvers"
    for (name, solver) in [
         ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver=:cgls; kwargs...)),
         ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant=:Newton; kwargs...))
        ]
        @info "    $name"
        bound_constrained_nls(solver)
        multiprecision_nls(solver, :unc)
        multiprecision_nls(solver, :bnd)
    end
end

tests()

@info "Specific solver tests"
include("solvers/trunkls.jl")
include("incompatible.jl")
