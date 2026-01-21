# [JSOSolvers.jl documentation](@id Home)

`JSOSolvers.jl` is a collection of Julia optimization solvers for nonlinear, potentially nonconvex, continuous optimization problems that are unconstrained or bound-constrained:

```math
\begin{aligned}
    \min\; & f(x) \\
    \text{s.t.}\; & \ell \leq x \leq u
\end{aligned}
```
where $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is a continuously differentiable function, with  $\ell \in \left(\mathbb{R} \cup \{-\infty\} \right)^n$, and  $u \in \left(\mathbb{R} \cup \{+\infty\} \right)^n$.
The algorithms implemented here are iterative methods that aim to compute a stationary point of \eqref{eq:nlp} using first and, if possible, second-order derivatives.

This package provides optimization solvers curated by the [JuliaSmoothOptimizers](https://jso.dev) organization.
Solvers in `JSOSolvers.jl` take as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl`, a flexible data type to evaluate objective and constraints, their derivatives, and to provide any information that a solver might request from a model.

The solvers in `JSOSolvers.jl` adopt a matrix-free approach, where standard optimization methods are implemented without forming derivative matrices explicitly.
This strategy enables the solution of large-scale problems even when function and gradient evaluations are expensive. The motivation is to solve large-scale unconstrained and bound-constrained problems such as parameter estimation in inverse problems, design optimization in engineering, and regularized machine learning models, and use these solvers to solve subproblems of penalty algorithms.

## Installation

`JSOSolvers` is a registered package. To install this package, open the Julia REPL (i.e., execute the julia binary), type `]` to enter package mode, and install `JSOSolvers` as follows

```julia
pkg> add JSOSolvers
```

## Tests

You can run the package’s unit tests with:

```julia
pkg> test JSOSolvers
```

The test suite is run automatically in continuous integration for every pull request and release, and can also be executed locally as above.

### What the tests cover

The test suite is composed of two layers:
1. Unit tests and regression tests (solver-specific);
1. Convergence tests using SolverTest.jl (problem-specific);

#### Unit and regression tests
Together, these ensure correctness, robustness, API stability, and compatibility with the JSO ecosystem.

The tests cover, for all solvers:
- Allocation behaviour of the implemented solvers (to detect unnecessary allocations and regressions);
- The callback parameter and how it is invoked during the iterations;
- Consistency of keyword arguments, to confirm that all solvers accept a common base set of options;
- Proper error handling when an incompatible problem is passed;
- Special options in solvers, such as `use_only_objgrad` for TRON;
- Restart behaviour, i.e., re-solving a problem from a different initial guess while reusing previously allocated memory where possible;
- GPU tests, to ensure that solvers work correctly with GPU-backed arrays when supported.

These unit tests also act as regression tests: they serve to detect accidental behavioural changes when refactoring, updating dependencies, or extending solver capabilities.

#### Convergence tests

We use `SolverTest.jl` to run solvers on a collection of small analytic academic problems (unconstrained and bound-constrained) with different precision settings.
These are black-box tests focused solely on:
- reaching a first-order stationary point,
- producing a valid solver status,
- and respecting tolerances on optimality and feasibility.

They do not enforce a specific iteration path or number of iterations; they only ensure that solver outputs remain correct and stable across releases.

## Dependencies

`JSOSolvers.jl` is built on the JuliaSmoothOptimizers ecosystem and relies on a small set of packages to provide modeling, linear algebra utilities, and solver infrastructure.

The main dependencies required to use the solvers are:

- `NLPModels.jl` – defines the problem API used by all solvers;

- `SolverTools.jl` – provides common optimization components such as line searches, stopping conditions, and trace utilities;

- `Krylov.jl` – used by second-order methods such as TRON and TRUNK for solving Newton or trust-region subproblems;

- `LinearOperators.jl` – provides abstractions for matrix-free linear operators used by iterative methods;

- `SolverCore.jl` – provides the solver interface infrastructure;

- `NLPModelsModifiers.jl` – provides utilities for modifying models (e.g., adding bounds or transformations).
These dependencies are installed automatically when JSOSolvers.jl is added via the Julia package manager, and no additional configuration is required.

## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.

## Basic usage

All solvers here are _JSO-Compliant_, in the sense that they accept NLPModels and return GenericExecutionStats.
This allows [benchmark them easily](https://jso.dev/tutorials/introduction-to-solverbenchmark/).
See the full list of [Solvers](@ref).

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
- `callback` is a function that is called at each iteration.
- `stats` is a `SolverTools.GenericExecutionStats` with the output of the solver.

### Callback

The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of the residual, for instance, the norm of the gradient for unconstrained problems;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
