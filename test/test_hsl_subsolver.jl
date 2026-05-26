using HSL_jll
using HSL

if LIBHSL_isfunctional()
  @testset "Testing HSL Subsolvers" begin
    for (name, mySolver) in [
      (
        "R2N_ma97",
        (nlp; kwargs...) -> R2N(nlp; subsolver = MA97R2NSubsolver, kwargs...),
      ),
      (
        "R2N_ma97_ag",
        (nlp; kwargs...) -> R2N(nlp; subsolver = MA97R2NSubsolver, npc_handler = :ag, kwargs...),
      ),
      # ma57
      (
        "R2N_ma57",
        (nlp; kwargs...) -> R2N(nlp; subsolver = MA57R2NSubsolver, kwargs...),
      ),
      (   
        "R2N_ma57_ag",
        (nlp; kwargs...) -> R2N(nlp; subsolver = MA57R2NSubsolver, npc_handler = :ag, kwargs...),
      ),  
    ]
      @testset "Testing solver: $name" begin
        f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
        nlp = ADNLPModel(f, [-1.2; 1.0])

        stats = mySolver(nlp)
        @test stats.status == :first_order
        @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
      end
    end
  end
else
  println("Skipping HSL subsolver tests; LIBHSL is not functional.")
end