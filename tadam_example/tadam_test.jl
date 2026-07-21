using Pkg
Pkg.activate("tadam_example")

Pkg.develop(path=".")
Pkg.add([
    "ADNLPModels",
    "Krylov",
    "LinearOperators",
    "NLPModels",
    "NLPModelsModifiers",
    "SolverCore",
    "Revise",
])

using Revise
using JSOSolvers
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 


# # 1. Define the Problem (Extended Rosenbrock)
# n = 30
# nlp = ADNLPModel(
#     x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
#     collect(1:n) ./ (n + 1),
#     name = "Extended Rosenbrock"
# )


n = 30
A = 10.0

# Rastrigin function: Non-convex, multimodal, fully separable.
# Global minimum at x = 0, f(x) = 0.
rastrigin(x) = A * n + sum(x[i]^2 - A * cos(2 * pi * x[i]) for i in 1:n)

# Initialize away from the global minimum to force traversal of local optima
x0 = fill(1.5, n)

nlp = ADNLPModel(rastrigin, x0, name="Rastrigin")

println("Problem: $(nlp.meta.name)")


TADAM_stats = tadam(nlp, max_iter = 1500, verbose = 1)
TADAM_stats_2 = tadam(nlp, max_iter = 1200, verbose =1 ,  θ2 = 200 )
lbfgs_stats = lbfgs(nlp, max_iter = 1000, verbose = 1)

println("\nStastics Summary:")
println(TADAM_stats)

println("\nStastics Summary:")
println(TADAM_stats_2)

println("\nStastics LBFGS Summary:")
println(lbfgs_stats)