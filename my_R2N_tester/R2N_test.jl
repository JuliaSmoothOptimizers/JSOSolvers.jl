using Revise
using JSOSolvers
using HSL
using Arpack, TSVD, GenericLinearAlgebra
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 
using Printf
using CUTEst 
using OptimizationProblems, OptimizationProblems.ADNLPProblems
# using CUTEst, Quadmath


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

# nlp = CUTEstModel("TOINTQOR")
# nlp from optimization_problems.jl
nlp = morebv()

println("Problem: $(nlp.meta.name)")
# cg
println("\nRunning R2N with CG subsolver and AG NPC handler...")
stats_cg_ag = R2N(nlp; verbose = 1,subsolver_verbose=0, max_iter = 180, subsolver = CGR2NSubsolver, npc_handler = :ag)

println("\nSubsolver = MinresQlp, NPC Handler = AG")
stats_minres_qlp =
  R2N(nlp; verbose = 1, max_iter =100,subsolver = MinresQlpR2NSubsolver, npc_handler = :ag, subsolver_verbose = 0)

println("\nRunning trunk ")
stats_trunk = trunk(nlp, verbose=1, max_iter=150)

println("\nSubsolver = shifted_lbfgs")
stats_shifted_lbfgs = R2N(
    LBFGSModel(nlp); 
    verbose = 10, 
    max_iter = 1500, 
    subsolver = ShiftedLBFGSSolver, 
    npc_handler = :ag, 
    subsolver_verbose = 0
)

println("\nStastics Summary:")
println("stats_cg_ag: ", stats_cg_ag)
println("stats_minres_qlp: ", stats_minres_qlp)
println("stats_trunk: ", stats_trunk)
println("stats_shifted_lbfgs: ", stats_shifted_lbfgs)