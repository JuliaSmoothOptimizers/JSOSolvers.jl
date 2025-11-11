using Revise
using JSOSolvers
using HSL
using NLPModels, ADNLPModels
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore 

# f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
# nlp = ADNLPModel(f, [-1.2; 1.0])
# stats = R2N(nlp, verbose = 1)
#   println("The status is: ", stats.status == :first_order)
#   println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))



# for s in (:R2NSolver,  :FoSolver, :TronSolver, :TrunkSolver)
for s in (:R2NSolver,  )
    println("\n\n\t\t===================================")
    println("============        Testing solver: $s            ==============\n\n")
    f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f, [-1.2; 1.0])
    solver = eval(s)(nlp)
    

    stats = GenericExecutionStats(nlp)
    stats = SolverCore.solve!(solver, nlp, stats, verbose = 1)
    println("The status is: ", stats.status == :first_order)
    println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))

    f2(x) = (x[1])^2 + 4 * (x[2] - x[1]^2)^2
    nlp = ADNLPModel(f2, [-1.2; 1.0])
  
    solver = eval(s)(nlp,subsolver= :cg) 
    
    SolverCore.reset!(solver, nlp)

    stats = GenericExecutionStats(nlp)
    stats = SolverCore.solve!(solver, nlp, stats, verbose = 50,  atol = 1e-10, rtol = 1e-10)
    println("The status is: ", stats.status == :first_order)
    println("The solution is approximately: ", isapprox(stats.solution, [0.0; 0.0], atol = 1e-6))
    print(stats.solution)

end


println("example of different negative curvature handling strategies with R2N solver")
n = 30
nlp= ADNLPModel(
      x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
      collect(1:n) ./ (n + 1),
      name = "Extended Rosenbrock";
    )

stats_armjio = R2N(nlp, verbose = 1, max_iter=5000, subsolver= :minres_qlp, npc_handler= :armijo)
stats_sigma = R2N(nlp, verbose = 10, max_iter=5000, subsolver= :minres_qlp, npc_handler= :sigma)
stats_prev = R2N(nlp, verbose = 10, max_iter=5000, subsolver= :minres_qlp, npc_handler= :prev)
stats_cp = R2N(nlp, verbose = 10, max_iter=5000, subsolver= :minres_qlp, npc_handler= :cp)  
stats_trunk = trunk(nlp, verbose = 10, max_iter=5000)
println("The stats after 5k iteration is")
println("Armijo: ", stats_armjio.status, " max_iter :" , stats_armjio.iter)
println("Sigma: ", stats_sigma.status, " max_iter :" , stats_sigma.iter)
println("Previous: ", stats_prev.status, " max_iter :" , stats_prev.iter)
println("Cauchy Point: ", stats_cp.status, " max_iter :" , stats_cp.iter)
println("Truncated: ", stats_trunk.status, " max_iter :" , stats_trunk.iter)
println("\n\n\t\t===================================")

# # # const npc_handler_allowed = [:armijo, :sigma, :prev, :cp]
# for mysub in [:cg, :cr, :minres, :minres_qlp]
#   println("\n\n\t\t===================================")
#   println("============        Testing subsolver: $mysub            ==============\n\n")

#   for (name, mySolver) in [
#     ("R2N_cg_cp", (nlp, ; kwargs...) -> R2N(nlp; subsolver = mysub, npc_handler = :cp, kwargs...)),
#     (
#       "R2N_cg_prev",
#       (nlp, ; kwargs...) -> R2N(nlp; subsolver = mysub, npc_handler = :prev, kwargs...),
#     ),
#     (
#       "R2N_cg_sigma",
#       (nlp, ; kwargs...) -> R2N(nlp; subsolver = mysub, npc_handler = :sigma, kwargs...),
#     ),
#     (
#       "R2N_cg_armijo",
#       (nlp, ; kwargs...) -> R2N(nlp; subsolver = mysub, npc_handler = :armijo, kwargs...),
#     ),
#   ]
#     println("Testing solver: $name,$mySolver")

#     println("===================================")
#     f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
#     nlp = ADNLPModel(f, [-1.2; 1.0])

#     stats = mySolver(nlp, verbose = 1, max_iter = 14)
#     println("The status is: ", stats.status == :first_order)
#     println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))
#     println("===================================")
#   end
# end

# # testing R2N defualt and testing different subsolvers Krylov
# for (name, mySolver) in [
#   ("R2N", (nlp, ; kwargs...) -> R2N(nlp; kwargs...)),
#   ("R2N_cg", (nlp, ; kwargs...) -> R2N(nlp, subsolver = :cg; kwargs...)),
#   ("R2N_cr", (nlp, ; kwargs...) -> R2N(nlp, subsolver = :cr; kwargs...)),
#   ("R2N_minres", (nlp, ; kwargs...) -> R2N(nlp, subsolver = :minres; kwargs...)),
#   ("R2N_minres_qlp", (nlp, ; kwargs...) -> R2N(nlp, subsolver = :minres_qlp; kwargs...)),
# ]
#   println("Testing solver: $name,$mySolver")

#   println("===================================")
#   f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
#   nlp = ADNLPModel(f, [-1.2; 1.0])

#   stats = mySolver(nlp, verbose = 1)
#   println("The status is: ", stats.status == :first_order)
#   println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))
# end

# # testing LBFGS and Krylov

# # TESITNG lbfgs EXACT SOLVER AS SUBSOLVER
# for (name, mySolver) in [
#   ("R2N_lbfgs", (nlp, ; kwargs...) -> R2N(LBFGSModel(nlp), subsolver = :shifted_lbfgs; kwargs...)),
# ]
#   println("Testing solver: $name,$mySolver")

#   println("===================================")
#   f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
#   nlp = ADNLPModel(f, [-1.2; 1.0])

#   stats = mySolver(nlp, verbose = 1)
#   println("The status is: ", stats.status == :first_order)
#   println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))
# end

# # testing MA97
# if LIBHSL_isfunctional() #TODO
#   f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
#   nlp = ADNLPModel(f, [-1.2; 1.0])

#   stats = GenericExecutionStats(nlp)

#   solver = R2NSolver(nlp, subsolver = :ma97)
#   stats = SolverCore.solve!(solver, nlp, stats, verbose = 1, max_iter = 50)
#   println("The status is: ", stats.status == :first_order)
#   println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))
# else
#   @warn("HSL library is not functional. Skipping MA97 tests.")
# end

# # testing MA57
# if LIBHSL_isfunctional() #TODO
#   f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
#   nlp = ADNLPModel(f, [-1.2; 1.0])

#   stats = GenericExecutionStats(nlp)

#   solver = R2NSolver(nlp, subsolver = :ma57)
#   stats = SolverCore.solve!(solver, nlp, stats, verbose = 1, max_iter = 50)
#   println("The status is: ", stats.status == :first_order)
#   println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))
# else
#   @warn("HSL library is not functional. Skipping MA57 tests.")
# end

# # testing different -ve handling strategies with Krylov solvers 
