using Revise
using JSOSolvers
using HSL
using Arpack, TSVD, GenericLinearAlgebra
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 
using Printf

println("==============================================================")
println("      Testing R2N with different NPC Handling Strategies      ")
println("==============================================================")

# 1. Define the Problem (Extended Rosenbrock)
n = 30
nlp = ADNLPModel(
    x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
    collect(1:n) ./ (n + 1),
    name = "Extended Rosenbrock"
)

# 2. Define Solver Configurations

# List of strategies to test
solvers_to_test = [
    ("GS (Goldstein)",   MinresQlpR2NSubsolver, :gs,    0.0),
    ("Sigma Increase",   MinresQlpR2NSubsolver, :sigma, 1.0),
    ("Previous Step",    MinresQlpR2NSubsolver, :prev,  0.0),
    ("Cauchy Point",     MinresQlpR2NSubsolver, :cp,    1.0),
]

# 3. Run R2N Variants
results = []

for (name, sub_type, handler, sigma_min) in solvers_to_test
    println("\nRunning $name...")
    stats = R2N(
        nlp; 
        verbose = 5, 
        max_iter = 700, 
        subsolver = sub_type, 
        npc_handler = handler,
        σmin = sigma_min
    )
    push!(results, (name, stats))
end

# 4. Run Benchmark Solver (Trunk from JSOSolvers)
println("\nRunning Trunk (JSOSolvers)...")
stats_trunk = trunk(nlp; verbose = 1, max_iter = 700)
push!(results, ("Trunk", stats_trunk))


# 5. Print Summary Table
println("\n\n")
println("==============================================================")
println("                     Benchmark Results                        ")
println("==============================================================")
@printf("%-20s %-15s %-10s %-15s\n", "Strategy", "Status", "Iter", "Final Obj")
println("--------------------------------------------------------------")

for (name, st) in results
    @printf("%-20s %-15s %-10d %-15.4e\n", 
        name, st.status, st.iter, st.objective)
end
println("==============================================================\n")


# ==============================================================================
#  HSL Specific Unit Tests
# ==============================================================================

if LIBHSL_isfunctional()
    println("\nRunning HSL Unit Tests (MA97 & MA57)...")
    
    # Simple 2D Rosenbrock for unit testing
    f_test(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp_test = ADNLPModel(f_test, [-1.2; 1.0])

    for (sub_name, sub_type) in [("MA97", MA97R2NSubsolver), ("MA57", MA57R2NSubsolver)]
        println("  Testing $sub_name...")
        
        # Reset problem
        stats_test = R2N(
            nlp_test; 
            subsolver = sub_type, 
            verbose = 0, 
            max_iter = 50
        )
        
        is_solved = stats_test.status == :first_order
        is_accurate = isapprox(stats_test.solution, [1.0; 1.0], atol = 1e-6)
        
        status_msg = (is_solved && is_accurate) ? "✅ PASSED" : "❌ FAILED"
        println("  -> $status_msg (Iter: $(stats_test.iter))")
    end
else
    @warn "HSL library is not functional. Skipping HSL unit tests."
end