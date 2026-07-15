using HSL_jll
using HSL
if LIBHSL_isfunctional()
  @testset "Testing HSL Subsolvers & Memory Safety" begin
    
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