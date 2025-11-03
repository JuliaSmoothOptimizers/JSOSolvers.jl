using Revise
using JSOSolvers
using HSL
using NLPModels, ADNLPModels
using SparseArrays, LinearAlgebra
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])
stats = R2N(nlp, verbose = 1)
  println("The status is: ", stats.status == :first_order)
  println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))

s = :R2NSolver
fun = :R2N
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])
# nlp.meta.x0 .= 2.0


if fun == :R2N_exact
    nlp = LBFGSModel(nlp)
    solver = eval(s)(nlp,subsolver= :shifted_lbfgs)
  else 
    solver = eval(s)(nlp)
  end

  stats = GenericExecutionStats(nlp)
  stats = SolverCore.solve!(solver, nlp, stats, verbose = 1)
  println("The status is: ", stats.status == :first_order)
  println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))

  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

    stats = GenericExecutionStats(nlp)
  stats = SolverCore.solve!(solver, nlp, stats, verbose = 1)
  println("The status is: ", stats.status == :first_order)
  println("The solution is approximately: ", isapprox(stats.solution, [1.0; 1.0], atol = 1e-6))



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

# # testing different -ve handling strategies with Krylov solvers 
