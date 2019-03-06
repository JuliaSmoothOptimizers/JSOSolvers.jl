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
