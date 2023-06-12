using SolverTest

function tests()
  @testset "Testing NLP solvers" begin
    @testset "Unconstrained solvers" begin
      @testset "$name" for (name, solver) in [
        ("trunk+cg", (nlp; kwargs...) -> trunk(nlp, subsolver_type = CgSolver; kwargs...)),
        ("lbfgs", lbfgs),
        ("tron", tron),
        ("R2", R2),
      ]
        unconstrained_nlp(solver)
        multiprecision_nlp(solver, :unc)
      end
    end
    @testset "Bound-constrained solvers" begin
      @testset "$solver" for solver in [tron]
        bound_constrained_nlp(solver)
        multiprecision_nlp(solver, :unc)
        multiprecision_nlp(solver, :bnd)
      end
    end
  end
  @testset "Testing NLS solvers" begin
    @testset "Unconstrained solvers" begin
      @testset "$name" for (name, solver) in [
        ("trunk+cgls", (nls; kwargs...) -> trunk(nls, subsolver_type = CglsSolver; kwargs...)), # trunk with cgls due to multiprecision
        ("trunk full Hessian", (nls; kwargs...) -> trunk(nls, variant = :Newton; kwargs...)),
        ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver_type = CglsSolver; kwargs...)),
        ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant = :Newton; kwargs...)),
      ]
        unconstrained_nls(solver)
        multiprecision_nls(solver, :unc)
      end
    end
    @testset "Bound-constrained solvers" begin
      @testset "$name" for (name, solver) in [
        ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver_type = CglsSolver; kwargs...)),
        ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant = :Newton; kwargs...)),
      ]
        bound_constrained_nls(solver)
        multiprecision_nls(solver, :unc)
        multiprecision_nls(solver, :bnd)
      end
    end
  end
end

tests()

include("solvers/trunkls.jl")
include("incompatible.jl")
