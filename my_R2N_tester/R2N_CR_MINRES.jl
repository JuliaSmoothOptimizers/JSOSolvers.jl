using Revise
using JSOSolvers
using HSL
using Arpack, TSVD, GenericLinearAlgebra
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 
using Printf
using CUTEst 
# 1. Define the Problem
n = 30
nlp = ADNLPModel(
  x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
  collect(1:n) ./ (n + 1),
  name = "Extended Rosenbrock",
)

# 2. Define Parameter Grids for Combinations
# subsolvers = [CGR2NSubsolver, CRR2NSubsolver, MinresR2NSubsolver, MinresQlpR2NSubsolver]

println("Subsolver = CG, NPC Handler = AG")
stats_cg = R2N(nlp; verbose = 1, max_iter = 200, subsolver = CGR2NSubsolver, npc_handler = :ag)

println("\nSubsolver = Minres, NPC Handler = AG")
stats_minres =
  R2N(nlp; verbose = 1, max_iter = 50, subsolver = MinresR2NSubsolver, npc_handler = :ag, subsolver_verbose = 0)

println("\nSubsolver = CR, NPC Handler = AG")
stats_cr = R2N(nlp; verbose = 1, max_iter = 200, subsolver = CRR2NSubsolver, npc_handler = :ag)

println("\nSubsolver = MinresQlp, NPC Handler = AG")
stats_minres_qlp =
  R2N(nlp; verbose = 1, max_iter = 50, subsolver = MinresQlpR2NSubsolver, npc_handler = :ag, subsolver_verbose = 0)


println("\nStastics Summary:")
println("stats_minres: ", stats_minres)
println("stats_cg: ", stats_cg)
println("stats_cr: ", stats_cr)
println("stats_minres_qlp: ", stats_minres_qlp)