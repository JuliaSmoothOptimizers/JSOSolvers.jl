using Revise
using JSOSolvers
using HSL
using Arpack, TSVD, GenericLinearAlgebra
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 
using Printf
using CUTEst 

using OptimizationProblems, OptimizationProblems.ADNLPProblems
nlp  =  chainwoo(n=100)

solvers_to_test = [
    ("GS (Goldstein)--minres",   MinresR2NSubsolver, :ag,    0.0),
    ("GS (Goldstein)--cr",   CRR2NSubsolver, :ag,    0.0),
    # ("Sigma Increase",   MinresR2NSubsolver, :sigma, 0.0),
    # ("Previous Step",    MinresR2NSubsolver, :prev,  0.0),
    # ("Cauchy Point",     MinresR2NSubsolver, :cp,    0.0),
]


# 3. Run R2N Variants
results = []

for (name, sub_type, handler, sigma_min) in solvers_to_test
    println("\nRunning $name...")
    stats = R2N(
        nlp; 
        verbose = 1, 
        max_iter = 700, 
        subsolver = sub_type, 
        npc_handler = handler,
        σmin = sigma_min
    )
    push!(results, (name, stats))
    print("stats: ")
    print(stats)
end

# 4. Run Benchmark Solver (Trunk from JSOSolvers)
println("\nRunning Trunk (JSOSolvers)...")
stats_trunk = trunk(nlp; verbose = 1, max_iter = 700)
push!(results, ("Trunk", stats_trunk))


#####
# LBFGS
# 1. Wrap the model so R2N knows it's a Quasi-Newton problem
qn_nlp = LBFGSModel(nlp)

R2N(
    qn_nlp; 
    subsolver = MinresR2NSubsolver, # Just pass the type, R2N will construct it with qn_nlp
    npc_handler = :ag, 
    max_iter = 750, 
    η1 = 1.0e-6, 
    verbose = 10, 
    fast_local_convergence = false
)



qn_nlp = LBFGSModel(nlp)

R2N(
    qn_nlp; 
    subsolver = ShiftedLBFGSSolver, # Just pass the type, R2N will construct it with qn_nlp
    npc_handler = :ag, 
    max_iter = 750, 
    η1 = 1.0e-6, 
    verbose = 10, 
    fast_local_convergence = false
)



