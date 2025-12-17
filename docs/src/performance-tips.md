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
