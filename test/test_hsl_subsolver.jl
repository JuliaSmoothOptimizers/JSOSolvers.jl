using HSL_jll
using HSL
if LIBHSL_isfunctional()
    @testset "Testing HSL Subsolvers" begin
        for (name, mySolver) in [
            (
            "R2N_ma97",
            (nlp; kwargs...) -> R2N(nlp; subsolver = :ma97, kwargs...),
            ),
            (
            "R2N_ma97_armijo",
            (nlp; kwargs...) -> R2N(nlp; subsolver = :ma97, npc_handler = :armijo, kwargs...),
            ),
            # ma57
            (
            "R2N_ma57",
            (nlp; kwargs...) -> R2N(nlp; subsolver = :ma57, kwargs...),
            ),
            (   
            "R2N_ma57_armijo",
            (nlp; kwargs...) -> R2N(nlp; subsolver = :ma57, npc_handler = :armijo, kwargs...),
            ),  
        ]
            @testset "Testing solver: $name" begin
                f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
                nlp = ADNLPModel(f, [-1.2; 1.0])
            
                stats = mySolver(nlp, verbose = 1, max_iter = 14)
                @test stats.status == :first_order "Solver $name did not converge to first-order optimality."
                @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6) "Solver $name did not find the correct solution."
            end
        end
    end
else
    println("Skipping HSL subsolver tests; LIBHSL is not functional.")
end