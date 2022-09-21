# JSOSolvers.jl Tutorial

This package provides optimization solvers curated by the
[JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io)
organization.
All solvers are based on [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl).

This package contains the implementation of four algorithms that are classical for unconstrained/bound-constrained nonlinear optimization:
`lbfgs`, `R2`, `tron`, and `trunk`.

## Contents

```@contents
Pages = ["tutorial.md"]
```

## Solver input and output

All solvers have the following signature:

```
    stats = name_solver(nlp; kwargs...)
```

where `name_solver` can be `lbfgs`, `R2`, `tron`, or `trunk`, and with
- `nlp::AbstractNLPModel{T, V}` is an AbstractNLPModel or some specialization, such as an `AbstractNLSModel`;
- `stats::GenericExecutionStats{T, V}` is a `GenericExecutionStats`, see `SolverCore.jl`.

The keyword arguments may include:
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.

Refer to the documentation of each solver for further details on the available keyword arguments.

## Specialization for nonlinear least-squares

The solvers `tron` and `trunk` both have a specialized implementation for input models of type `AbstractNLSModel`.

The following example illustrate this specialization.

```@example
using JSOSolvers, ADNLPModels
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])
trunk(nlp, verbose = 1)
```

```@example
nlp.counters
```

```@example
F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
nls = ADNLSModel(F, [-1.2; 1.0], 2)
trunk(nls, verbose = 1)
```

```@example
nls.counters
```

## Advanced usage

For advanced usage, first define a `Solver` structure to preallocate the memory used in the algorithm, and then call `solve!`.

```
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
solver = LBFGSSolver(nlp; mem = 5);
stats = solve!(solver, nlp)
```

The following table provides the correspondance between the solvers and the solvers structures:
| Algorithm           | Solver structure |
| ------------------- | ---------------- |
| lbfgs               | LBFGSSolver      |
| R2                  | R2Solver         |
| tron                | TronSolver       |
| trunk               | TrunkSolver      |
| tron (nls-variant)  | TronSolverNLS    |
| trunk (nls-variant) | TrunkSolverNLS   |

It is also possible to pre-allocate the output structure `stats` and call `solve!(solver, nlp, stats)`.
```
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
solver = LBFGSSolver(nlp; mem = 5);
stats = GenericExecutionStats(nlp)
solve!(solver, nlp, stats)
```

## Callback

All the solvers have a callback mechanism called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver` see the documentation of each solver for details.

Below you can see an example of execution of the solver `trunk` with a callback to plot the iterates and create an animation.

```@example
using ADNLPModels, JSOSolvers, LinearAlgebra, Logging, Plots
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])

X = [nlp.meta.x0[1]]
Y = [nlp.meta.x0[2]]
function cb(nlp, solver, stats)
  x = solver.x
  push!(X, x[1])
  push!(Y, x[2])
  if solver.output.iter == 4
    solver.output.status = :user
  end
end
stats = trunk(nlp, callback=cb)
```

```@example
plot(leg=false)
xg = range(-1.5, 1.5, length=50)
yg = range(-1.5, 1.5, length=50)
contour!(xg, yg, (x1,x2) -> f([x1; x2]), levels=100)
plot!(X, Y, c=:red, l=:arrow, m=4)
```
