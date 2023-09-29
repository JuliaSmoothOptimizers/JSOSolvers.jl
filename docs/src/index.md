# [JSOSolvers.jl documentation](@id Home)

This package provides a few optimization solvers curated by the [JuliaSmoothOptimizers](https://jso.dev) organization.

## Basic usage

All solvers here are _JSO-Compliant_, in the sense that they accept NLPModels and return GenericExecutionStats.
This allows [benchmark them easily](https://jso.dev/tutorials/introduction-to-solverbenchmark/).

All solvers can be called like the following:

```julia
stats = solver_name(nlp; kwargs...)
```

where `nlp` is an AbstractNLPModel or some specialization, such as an `AbstractNLSModel`, and the following keyword arguments are supported:

- `x` is the starting default (default: `nlp.meta.x0`);
- `atol` is the absolute stopping tolerance (default: `atol = √ϵ`);
- `rtol` is the relative stopping tolerance (default: `rtol = √ϵ`);
- `max_eval` is the maximum number of objective and constraints function evaluations (default: `-1`, which means no limit);
- `max_time` is the maximum allowed elapsed time (default: `30.0`);
- `stats` is a `SolverTools.GenericExecutionStats` with the output of the solver.

See the full list of [Solvers](@ref).
