# [JSOSolvers.jl documentation](@id Home)

This package provides optimization solvers curated by the
[JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io)
organization.
All solvers are based on [SolverTools.jl](https://github.com/JuliaSmoothOptimizers/SolverTools.jl).

TODO:
- [ ] Show benchmarks of these solvers (using
  [SolverBenchmark.jl](https://github.com/JuliaSmoothOptimizers/SolverBenchmark.jl));
- [ ] Profile solver to identify bottlenecks;
- [ ] Use `PkgBenchmark` and `SolverBenchmark` to compare future PRs.

## Solver input and output

All solvers use the input and output given by `SolverTools`. Every solver has the
following call signature:

```
stats = solver(nlp; x, atol, rtol, max_eval, max_time, ...)
```

where
- `nlp` is an AbstractNLPModel or some specialization, such as an `AbstractNLSModel`;
- `x` is the starting default (default: `nlp.meta.x0`);
- `atol` is the absolute stopping tolerance (default: `atol = √ϵ`);
- `rtol` is the relative stopping tolerance (default: `rtol = √ϵ`);
- `max_eval` is the maximum number of objective and constraints function evaluations (default: `-1`, which means no limit);
- `max_time` is the maximum allowed elapsed time (default: `30.0`);
- `stats` is a `SolverTools.GenericExecutionStats` with the output of the solver.

## Callback

Below you can see an example of execution of the solver `trunk` with a callback to plot the iterations and create an animation.

```@example
using ADNLPModels, JSOSolvers, LinearAlgebra, Logging, Plots

function callback_example()
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  xg = range(-1.5, 1.5, length=50)
  yg = range(-1.5, 1.5, length=50)

  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]

  function cb(nlp, solver)
    x = solver.x
    push!(X, x[1])
    push!(Y, x[2])

    if solver.output.iter == 4
      solver.output.status = :user
    end
  end

  output = with_logger(NullLogger()) do
    trunk(nlp, callback=cb)
  end

  plot(leg=false)
  contour!(xg, yg, (x1,x2) -> f([x1; x2]), levels=100)
  plot!(X, Y, c=:red, l=:arrow, m=4)
end

callback_example()
```