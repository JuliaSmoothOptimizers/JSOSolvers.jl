---
title: 'JSOSolvers.jl: JuliaSmoothOptimizers optimization solvers'
tags:
  - Julia
  - nonlinear optimization
  - numerical optimization
  - large-scale optimization
  - unconstrained optimization
authors:
  - name: Tangi Migot^[corresponding author]
    orcid: 0000-0001-7729-2513
    affiliation: 1
  - name: Dominique Monnet
    orcid:
    affiliation: 3
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
  - name: Abel Soares Siqueira
    orcid: 0000-0003-4451-281X
    affiliation: 2
affiliations:
  - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
    index: 1
  - name: Netherlands eScience Center, Amsterdam, NL
    index: 2
  - name: Univ Rennes, INSA Rennes, CNRS, IRMAR - UMR 6625, Rennes, France
    index: 3
date: 25 September 2024
bibliography: paper.bib

---

# Summary

`JSOSolvers.jl` is a Julia [@bezanson2017julia] implementation of optimization solvers to address nonlinear nonconvex continuous optimization problems that are unconstrained or bound-constrained
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \ f(x) \quad \text{subject to} \quad \ell \leq x \leq u,
\end{equation}
where  $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is (twice) continuously differentiable functions, with  $\ell \in \left(\mathbb{R} \cup \{-\infty\} \right)^n$, and  $u \in \left(\mathbb{R} \cup \{+\infty\} \right)^n$.

The algorithms implemented here are iterative methods that aim to compute a local minimum of \eqref{eq:nlp} using first and if possible second-order derivatives.

Our initial motivation for considering \eqref{eq:nlp} and developing `JSOSolvers.jl` is to solve large-scale unconstrained and bound-constrained problems such as ......, and use these solvers to solve subproblems of penalty algorithms such `Percival.jl`[@Percival_jl] or `FletcherPenaltySolver.jl` [@FletcherPenaltySolver_jl].
In many of such engineering design problems, it is not computationally feasible or realistic to store Jacobians or Hessians explicitly.
Matrix-free implementations of standard optimization methods—implementations that do not explicitly form Jacobians and Hessians.
The matrix-free approach makes solving problems with thousands of design variables and constraints tractable, even if function and gradient evaluations are costly.

*
Expliquer les différents algos et leurs implémentations
TRON: A trust-region solver for bound-constrained problems, implementing  [@tron];
TRUNK: A factorization-free trust-region truncated-CG Newton method following the description given in [@conn2000trust];
L-BFGS: A limited-memory factorization-free line-search BFGS method;
FOMO: a first-order quadratic regularization method.
The implementation of TRON and TRUNK also have nonlinear least squares variants.

List of strenght of JSO-Solvers:
in-place solve with 0 allocations
flexible data type
some of the solvers are GPU-compatible
*

`JSOSolvers.jl` is built upon the JuliaSmoothOptimizers (JSO) tools [@jso].
JSO is an academic organization containing a collection of Julia packages for nonlinear optimization software development, testing, and benchmarking.
It provides tools for building models, accessing problems repositories, and solving subproblems.
`JSOSolvers.jl` takes as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl` [@NLPModels_jl], a flexible data type to evaluate objective and constraints, their derivatives, and to provide any information that a solver might request from a model.
The user can hand-code derivatives, use automatic differentiation, or use JSO-interfaces to classical mathematical optimization modeling languages such as AMPL [@fourer2003ampl], CUTEst [@cutest], or JuMP [@jump]. 

These solvers relies heavily on iterative linear algebra methods from `Krylov.jl` [@Krylov_jl], which provides more than 35 implementations of standard and novel Krylov methods.
Notably, Krylov.jl support various floating-point systems compatible with Julia and provides GPU acceleration through backends that build on GPUArrays.jl [@GPUArrays], including CUDA.jl [@besard2018juliagpu], AMDGPU.jl [@amdgpu], and oneAPI.jl [@oneAPI], the Julia interfaces to NVIDIA, AMD, and Intel GPUs.

# Statement of need

Julia's JIT compiler is attractive for the design of efficient scientific computing software, and, in particular, mathematical optimization [@lubin2015computing], and has become a natural choice for developing new solvers.

There already exist ways to solve \eqref{eq:nlp} in Julia.
If \eqref{eq:nlp} is amenable to being modeled in `JuMP` [@jump], the model may be passed to state-of-the-art solvers, implemented in low-level compiled languages, via wrappers thanks to Julia's native interoperability with such languages.
However, interfaces to low-level languages have limitations that pure Julia implementations do not have, including the ability to apply solvers with various arithmetic types.

Although most personal computers offer IEEE 754 single and double precision computations,
new architectures implement native computations in other floating-point systems.  In addition,
software libraries such as the GNU MPFR, shipped with Julia, let users experiment with
computations in variable, extended precision at the software level with the `BigFloat` data type.
Working in high precision has obvious benefits in terms of accuracy.

Alternatives:
- `Optim.jl` [@mogensen2018optim] implements a ...
- GALAHAD.jl https://github.com/ralna/GALAHAD
- Pure Julia alternatives `AdaptiveRegularization.jl` [@AdaptiveRegularization_jl]

## Support for any floating-point system supported by Julia

*Give an exampls here? A GPU example? Probably in the documentation instead of here.*

## In-place methods

* This should be in the documentation not here. *

The main functions have an in-place variant that allows to solve multiple optimization problems with the same dimensions, precision and architecture. More complex solvers can take advantage of this functionality by allocating workspace for the solve only once. The in-place variants only require a Julia structure that contains all the storage needed as additional argument. In-place methods limit memory allocations and deallocations, which are particularly expensive on GPUs.

```
using NLPModels, NLPModelsTest
nlp = HS6() # test problem from NLPModelsTest whose evaluations are allocation free
using SolverCore
stats = GenericExecutionStats(nlp) # pre-allocate output structure
using JSOSolvers
solver = TronSolver(nlp) # pre-allocate workspace
SolverCore.solve!(solver, nlp, stats) # call percival
JSOSolvers.reset!(solver)
NLPModels.reset!(nlp)
@allocated SolverCore.solve!(solver, nlp, stats) == 0
```

## Numerics

`JSOSolvers.jl` can solve large-scale problems and can be benchmarked easily against other JSO-compliant solvers using `SolverBenchmark.jl` [@SolverBenchmark_jl].
We include below performance profiles [@dolan2002benchmarking] of `JSOSolvers.jl` against Ipopt on all problems from CUTEst [@cutest]. 

<!--
illustrating that `Percival` is a fast and stable alternative to a state of the art solver

NOTE: Putting the code is too long
```
include("make_problems_list.jl") # setup a file `list_problems.dat` with problem names
include("benchmark.jl") # run the benchmark and store the result in `ipopt_percival_82.jld2`
include("figures.jl") # make the figure
```

![](ipopt_percival_96.png){ width=100% }
-->

# Acknowledgements

Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
