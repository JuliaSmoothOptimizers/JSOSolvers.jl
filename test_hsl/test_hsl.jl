# using Pkg
# Pkg.activate("hsl_example")

using Revise
using JSOSolvers
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore  
using HSL
using Arpack, TSVD, GenericLinearAlgebra
using SparseArrays, LinearAlgebra
using OptimizationProblems, OptimizationProblems.ADNLPProblems

T = Float64
ATOL = 1.0e-6
RTOL = 1.0e-6

println("===================================== freuroth =====================================")
nlp = freuroth(type = T ,n = 100)
stats =  R2N(nlp; subsolver=MA57R2NSubsolver(nlp), npc_handler=:ag,  verbose=1, rtol=RTOL, atol=ATOL)

println("===================================== genhumps_WITH_MINRES =====================================")
nlp = genhumps(type = T, n = 100)
stats =  R2N(nlp; subsolver=MinresQlpR2NSubsolver, npc_handler=:ag,  verbose=100, rtol=RTOL, atol=ATOL)

print("Stats: ", stats)
# println("===================================== genhumps_WITH_MINRES_sigma =====================================")
# nlp = genhumps(type = T, n = 100)
# stats =  R2N(nlp; subsolver=MinresQlpR2NSubsolver(nlp), npc_handler=:sigma,  verbose=100, rtol=RTOL, atol=ATOL, maxiter=1000)
# print("Stats: ", stats)


println("===================================== genhumps =====================================")
nlp = genhumps(type = T, n = 100)
stats =  R2N(nlp; subsolver=MA57R2NSubsolver(nlp), npc_handler=:ag,  verbose=100, rtol=RTOL, atol=ATOL)
print("Stats: ", stats)


println("===================================== genhumps_γ2 =====================================")
nlp = genhumps(type = T, n = 100)
stats =  R2N(nlp; subsolver=MA57R2NSubsolver(nlp), npc_handler=:ag, γ2=100.0 ,verbose=100, rtol=RTOL, atol=ATOL)
print("Stats: ", stats)

# println("===================================== indef_mod =====================================")
# nlp = genrose(type = T ,n = 100)
# stats =  indef_mod(nlp; subsolver=MA57R2NSubsolver(nlp), npc_handler=:ag,  verbose=1, rtol=RTOL, atol=ATOL)

# # println("===================================== genrose_nash =====================================")
# # nlp = genrose_nash(type = T ,n = 100)