using HSL_jll
using HSL
@testset "HSL input-size guards" begin
  nlp = ADNLPModel(x -> sum(abs2, x), ones(2))
  for constructor in (MA57R2NSubsolver, MA97R2NSubsolver)
    @test_throws ArgumentError constructor(nlp; max_nvar = -1)
    @test_throws ArgumentError constructor(nlp; max_nnzh = -1)
    for limits in ((max_nvar = 1,), (max_nnzh = nlp.meta.nnzh - 1,))
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