using HSL_jll
using HSL
@testset "HSL input-size guards" begin
  nlp = ADNLPModel(x -> sum(abs2, x), ones(2))
  for constructor in (MA57R2NSubsolver, MA97R2NSubsolver)
    @test_throws ArgumentError constructor(nlp; max_nvar = -1)
    @test_throws ArgumentError constructor(nlp; max_nnzh = -1)
    @test_throws ArgumentError constructor(nlp; dense_max_nvar = -1)
    @test_throws ArgumentError constructor(nlp; min_matrix_size = -1)
    for limits in (
      (max_nvar = 1,),
      (max_nnzh = nlp.meta.nnzh - 1,),
      # dense 2x2 Hessian rejected only once the density guard is armed at n = 1
      (dense_max_nvar = 1,),
    )
      sub = constructor(nlp; limits...)
      @test JSOSolvers.is_unsupported(sub)
      @test sub.hsl_obj === nothing
      @test isempty(sub.rows) && isempty(sub.cols) && isempty(sub.vals)
      JSOSolvers.reset_subsolver!(sub, nlp, nlp.meta.x0)
      @test JSOSolvers.is_unsupported(sub)
      stats = @test_logs (:error, r"skipping HSL") R2N(nlp; subsolver = sub)
      @test stats.status == :exception
      @test stats.iter == 0
    end
  end
end

if LIBHSL_isfunctional()
  @testset "Testing HSL Subsolvers & Memory Safety" begin
    dense_nlp = ADNLPModel(x -> sum(abs2, x), ones(2))
    small_dense_subsolver = MA97R2NSubsolver(dense_nlp)
    @test !is_unsupported(small_dense_subsolver)
    finalize_subsolver!(small_dense_subsolver)
    @test is_unsupported(MA97R2NSubsolver(dense_nlp; min_matrix_size = 0))
    @test is_unsupported(MA97R2NSubsolver(dense_nlp; dense_max_nvar = 0))

    # `dense_max_nvar` is the largest `n` whose dense Hessian is still accepted:
    # the default keeps moderately sized dense problems (e.g. n = 5_000) on HSL
    # and only rejects the ones above it.
    @test JSOSolvers.dense_hessian_entries(5_000) == (5_000 * 5_001) ÷ 2
    for n in (2, 500, 5_000, JSOSolvers.DEFAULT_HSL_DENSE_MAX_NVAR)
      dense_nnzh = JSOSolvers.dense_hessian_entries(n)
      @test !JSOSolvers.hsl_guard_triggered(
        n,
        dense_nnzh,
        0.5,
        JSOSolvers.dense_hessian_entries(JSOSolvers.DEFAULT_HSL_DENSE_MAX_NVAR),
        typemax(Int),
        typemax(Int),
      )
    end
    let n = JSOSolvers.DEFAULT_HSL_DENSE_MAX_NVAR + 1
      min_size = JSOSolvers.dense_hessian_entries(JSOSolvers.DEFAULT_HSL_DENSE_MAX_NVAR)
      # dense above the threshold is rejected ...
      @test JSOSolvers.hsl_guard_triggered(
        n,
        JSOSolvers.dense_hessian_entries(n),
        0.5,
        min_size,
        typemax(Int),
        typemax(Int),
      )
      # ... but a genuinely sparse Hessian of the same size still goes to HSL
      @test !JSOSolvers.hsl_guard_triggered(n, 5n, 0.5, min_size, typemax(Int), typemax(Int))
    end

    # a dense problem below the threshold is actually solved, not skipped
    dense_n = 60
    dense_quad = ADNLPModel(x -> sum(abs2, x) + abs2(sum(x)), ones(dense_n))
    @test dense_quad.meta.nnzh > 0.5 * JSOSolvers.dense_hessian_entries(dense_n)
    dense_sub = MA57R2NSubsolver(dense_quad)
    @test !is_unsupported(dense_sub)
    dense_stats = R2N(dense_quad; subsolver = dense_sub)
    @test dense_stats.status == :first_order

    for (name, subsolver_constructor, extra_kwargs) in [
      ("R2N_ma97",    MA97R2NSubsolver, NamedTuple()),
      ("R2N_ma97_ag", MA97R2NSubsolver, (npc_handler = :ag,)),
      ("R2N_ma57",    MA57R2NSubsolver, NamedTuple()),
      ("R2N_ma57_ag", MA57R2NSubsolver, (npc_handler = :ag,)),
    ]
      @testset "Testing solver: $name" begin
        f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
        nlp = ADNLPModel(f, [-1.2; 1.0])

        sub_instance = subsolver_constructor(nlp)

        solver = R2NSolver(nlp; subsolver = sub_instance)
        
        stats = solve!(solver, nlp; extra_kwargs...)
        
        @test stats.status == :first_order
        @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

        # Crash Verification: Explicit Cleanup
        @test begin
          finalize(solver.subsolver)
          true
        end

        # Crash Verification: GC Trap (Double-Free Prevention)
        @test begin
          GC.gc()
          true
        end
      end
    end
  end
else
  println("Skipping HSL subsolver tests; LIBHSL is not functional.")
end