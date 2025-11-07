using SolverTest

function tests()
  @testset "Testing NLP solvers" begin
    @testset "Unconstrained solvers" begin
      @testset "$name" for (name, solver) in [
        ("trunk+cg", (nlp; kwargs...) -> trunk(nlp, subsolver = :cg; kwargs...)),
        ("lbfgs", lbfgs),
        ("tron", tron),
        ("R2", R2),
        ("fomo_r2_nesterov_HB", fomo),
        ("fomo_tr_nesterov_HB", (nlp; kwargs...) -> fomo(nlp, step_backend = JSOSolvers.tr_step(); kwargs...)),
        ("fomo_r2_cg_PR", (nlp; kwargs...) -> fomo(nlp, momentum_backend = JSOSolvers.cg_PR(); kwargs...)),
        ("fomo_tr_cg_PR", (nlp; kwargs...) -> fomo(nlp, step_backend = JSOSolvers.tr_step(), momentum_backend = JSOSolvers.cg_PR(); kwargs...)),
        ("fomo_r2_cg_FR", (nlp; kwargs...) -> fomo(nlp, momentum_backend = JSOSolvers.cg_FR(); kwargs...)),
        ("fomo_tr_cg_FR", (nlp; kwargs...) -> fomo(nlp, step_backend = JSOSolvers.tr_step(), momentum_backend = JSOSolvers.cg_FR(); kwargs...)),
      ]
        unconstrained_nlp(solver)
        multiprecision_nlp(solver, :unc)
      end
      @testset "$name : nonmonotone configuration" for (name, solver) in [
        ("R2", (nlp; kwargs...) -> R2(nlp, M = 2; kwargs...)),
        ("fomo_r2_nesterov_HB", (nlp; kwargs...) -> fomo(nlp, M = 2; kwargs...)),
        (
          "fomo_tr_nesterov_HB",
          (nlp; kwargs...) -> fomo(nlp, M = 2, step_backend = JSOSolvers.tr_step(); kwargs...),
        ),
        ("fomo_r2_cg_PR", (nlp; kwargs...) -> fomo(nlp, M = 2, momentum_backend = JSOSolvers.cg_PR(); kwargs...)),
        (
          "fomo_tr_cg_PR",
          (nlp; kwargs...) -> fomo(nlp, M = 2, step_backend = JSOSolvers.tr_step(), momentum_backend = JSOSolvers.cg_PR(); kwargs...),
        ),
        ("fomo_r2_cg_FR", (nlp; kwargs...) -> fomo(nlp, M = 2, momentum_backend = JSOSolvers.cg_FR(); kwargs...)),
        (
          "fomo_tr_cg_FR",
          (nlp; kwargs...) -> fomo(nlp, M = 2, step_backend = JSOSolvers.tr_step(), momentum_backend = JSOSolvers.cg_FR(); kwargs...),
        ),
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
        ("trunk+cgls", (nls; kwargs...) -> trunk(nls, subsolver = :cgls; kwargs...)), # trunk with cgls due to multiprecision
        ("trunk full Hessian", (nls; kwargs...) -> trunk(nls, variant = :Newton; kwargs...)),
        ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver = :cgls; kwargs...)),
        ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant = :Newton; kwargs...)),
      ]
        unconstrained_nls(solver)
        multiprecision_nls(solver, :unc)
      end
    end
    @testset "Bound-constrained solvers" begin
      @testset "$name" for (name, solver) in [
        ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver = :cgls; kwargs...)),
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
