# TODO: After all subsolvers handle multiprecision, drop this file and use
# `unconstrained-nls`
@testset "Simple test" begin
    model = ADNLSModel(x -> [10 * (x[2] - x[1]^2), 1 - x[1]], [-2.0, 1.0], 2)
    for subsolver in JSOSolvers.trunkls_allowed_subsolvers
        stats = trunk(model, subsolver=subsolver)
        @test stats.status == :first_order
        isapprox(stats.solution, ones(2), rtol=1e-4)
        @test isapprox(stats.objective, 0, atol=1e-6)
        @test neval_jac_residual(model) == 0
        stline = statsline(stats, [:objective, :dual_feas, :elapsed_time, :iter, :neval_residual,
                                   :neval_jprod_residual, :neval_jtprod_residual, :status])
        @info @sprintf("%8s  %5d  %5d  %s\n", model.meta.name, model.nls_meta.nvar, model.nls_meta.nequ, stline)
        reset!(model)
    end

    @test_throws ErrorException trunk(model, subsolver=:minres)
end

@testset "Larger test" begin
  n = 30
  model = ADNLSModel(x->[[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]], (1:n) ./ (n + 1), 2n - 2)
  for subsolver in JSOSolvers.trunkls_allowed_subsolvers
      stats = with_logger(NullLogger()) do
          trunk(model, subsolver=subsolver)
      end
      @test stats.status == :first_order
      @test isapprox(stats.objective, 0, atol=1e-6)
      @test neval_jac_residual(model) == 0
      stline = statsline(stats, [:objective, :dual_feas, :elapsed_time, :iter, :neval_residual,
                                 :neval_jprod_residual, :neval_jtprod_residual, :status])
      @info @sprintf("%8s  %5d  %5d  %s\n", model.meta.name, model.nls_meta.nvar, model.nls_meta.nequ, stline)
      reset!(model)
  end
end
