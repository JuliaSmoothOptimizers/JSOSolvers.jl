# Performance Tips

The main functions have an in-place variant that allows one to solve multiple optimization problems with the same dimensions and data types.

More general solvers can take advantage of this functionality by allocating workspace for the solve only once.
The in-place variants only require a Julia structure that contains all the storage needed as an additional argument.
In-place methods limit memory allocations and deallocations, which are particularly expensive on GPUs.

```@example
using NLPModels, NLPModelsTest
nlp = BROWNDEN() # test problem from NLPModelsTest with allocation free evaluations

using SolverCore
stats = GenericExecutionStats(nlp) # pre-allocate output structure
using JSOSolvers
solver = TronSolver(nlp) # pre-allocate workspace
SolverCore.solve!(solver, nlp, stats) # call tron in place
SolverCore.reset!(solver)
NLPModels.reset!(nlp) # reset counters
(@allocated SolverCore.solve!(solver, nlp, stats)) == 0

```

## Experimental: choosing the L-BFGS memory automatically

The memory parameter `mem` of [`lbfgs`](@ref) defaults to 5, but the best value depends on the problem.
With the optional packages LBFGSMemTuner.jl (not yet registered), Symbolics.jl (v6) and ExpressionTreeForge.jl loaded, `mem = :auto` asks a two-stage
statistical model (Gao, Migot and Mudalige, AMMCS 2026) to recommend a value:

```julia
using JSOSolvers, LBFGSMemTuner, Symbolics, ExpressionTreeForge, ADNLPModels
nlp = ADNLPModel(x -> sum((x[i] - i)^2 for i = 1:10) + sin(x[1]), ones(10))
stats = lbfgs(nlp; mem = :auto)
stats.solver_specific[:mem]  # the value that was used
```

Things to know before using it:

- The model was trained on scalable unconstrained problems of `OptimizationProblems.jl` with up to `10^5` variables. It is experimental and opt-in; the default stays `mem = 5`.
- It needs the symbolic objective (an `ADNLPModel`-like `nlp.f` that Symbolics.jl can trace); the expression-tree features are computed with ExpressionTreeForge.jl, as for the training data. Otherwise, or if `LBFGSMemTuner` is not loaded, `:auto` warns and falls back to the default.
- Extracting the problem features can cost more than the solve itself, so for repeated solves compute them once with `LBFGSMemTuner.extract_features(nlp)` and pass the recommended integer to `lbfgs`.
- The features include timings and allocations of the first objective/gradient call (as in training), so they depend on whether the functions are already compiled and on the machine; recommendations are most faithful in a fresh session.
