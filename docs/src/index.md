# [JSOSolvers.jl documentation](@id Home)

`JSOSolvers.jl` is a collection of Julia optimization solvers for nonlinear, potentially nonconvex, continuous optimization problems that are unconstrained or bound-constrained:

    min f(x)     s.t.  ℓ ≤ x ≤ u


where $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is a continuously differentiable function, with  $\ell \in \left(\mathbb{R} \cup \{-\infty\} \right)^n$, and  $u \in \left(\mathbb{R} \cup \{+\infty\} \right)^n$.
The algorithms implemented here are iterative methods that aim to compute a stationary point of \eqref{eq:nlp} using first and, if possible, second-order derivatives.

This package provides optimization solvers curated by the [JuliaSmoothOptimizers](https://jso.dev) organization.
Solvers in `JSOSolvers.jl` take as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl`, a flexible data type to evaluate objective and constraints, their derivatives, and to provide any information that a solver might request from a model.

The solvers in `JSOSolvers.jl` adopt a matrix-free approach, where standard optimization methods are implemented without forming derivative matrices explicitly.
This strategy enables the solution of large-scale problems even when function and gradient evaluations are expensive. The motivation is to solve large-scale unconstrained and bound-constrained problems such as parameter estimation in inverse problems, design optimization in engineering, and regularized machine learning models, and use these solvers to solve subproblems of penalty algorithms.

## Installation

`JSOSolvers` is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type `]` to enter package mode, and install `JSOSolvers` as follows

```julia
`pkg> add JSOSolvers`
```

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.

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
