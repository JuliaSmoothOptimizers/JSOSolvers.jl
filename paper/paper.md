---
title: 'JSOSolvers.jl: Unconstrained and bound-constrained optimization solvers'
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
  - name: Netherlands eScience Center, Amsterdam, Netherlands
    index: 2
  - name: Univ Rennes, INSA Rennes, CNRS, IRMAR - UMR 6625, Rennes, France
    index: 3
date: 18 October 2025
bibliography: paper.bib

---

# Summary

`JSOSolvers.jl` is a collection of Julia [@bezanson2017julia] optimization solvers for nonlinear, potentially nonconvex, continuous optimization problems that are unconstrained or bound-constrained:
\begin{equation}\label{eq:nlp}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \ f(x) \quad \text{subject to} \quad \ell \leq x \leq u,
\end{equation}
where  $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is a continuously differentiable function, with  $\ell \in \left(\mathbb{R} \cup \{-\infty\} \right)^n$, and  $u \in \left(\mathbb{R} \cup \{+\infty\} \right)^n$.
The algorithms implemented here are iterative methods that aim to compute a stationary point of \eqref{eq:nlp} using first and, if possible, second-order derivatives.

Our initial motivation for considering \eqref{eq:nlp} and developing `JSOSolvers.jl` is to solve large-scale unconstrained and bound-constrained problems such as parameter estimation in inverse problems, design optimization in engineering, and regularized machine learning models, and use these solvers to solve subproblems of penalty algorithms, such as `Percival.jl`[@Percival_jl] or `FletcherPenaltySolver.jl` [@FletcherPenaltySolver_jl], for constrained nonlinear continuous optimization problems.
In many of these problems, explicitly storing Hessian matrices is either computationally prohibitive or impractical.
The solvers in `JSOSolvers.jl` adopt a matrix-free approach, where standard optimization methods are implemented without forming derivative matrices explicitly.
This strategy enables the solution of large-scale problems even when function and gradient evaluations are expensive.

The library includes TRON, a trust-region Newton method for bound-constrained problems following the classical formulation of @tron, TRUNK, a factorization-free trust-region Newton method based on the truncated conjugate gradient method, as described by @conn2000trust, an implementation of L-BFGS, a limited-memory quasi-Newton method using a line search globalization strategy, and FOMO, a first-order method based on quadratic regularization designed for unconstrained optimization.
The latter is an extension of a quadratic regularization method described by @aravkin2022proximal, and called R2 in `JSOSolvers.jl`.
Unlike textbook implementations, our solvers introduce several design differences.
TRON operates in a factorization-free mode, while the original Fortran TRON requires an explicitly formed Hessian.
TRUNK departs from the Conn–Gould–Toint formulation by supporting a non-monotone mode and multiple subproblem solvers (CG, CR, MINRES, etc.).
Our L-BFGS implementation uses a simplified line-search strategy that avoids the standard Wolfe conditions while maintaining robust convergence in practice.

A nonlinear least-squares problem is a special case of \eqref{eq:nlp}, where $f(x)=\frac{1}{2}\|F(x)\|^2_2$ and the residual $F:\mathbb{R}^n \rightarrow \mathbb{R}^m$ is continuously differentiable, which appears in many applications, including inverse problems in imaging, geophysics, and machine learning. While it is possible to solve the problem using only the objective, knowing $F$ independently allows the development of more efficient methods.
TRON and TRUNK have specialized implementations leveraging the structure of residual models to improve performance and scalability.

A key strength of `JSOSolvers.jl` lies in its efficiency and flexibility.
The solvers support fully in-place execution, allowing repeated solves without additional memory allocation, which is particularly beneficial in high-performance and GPU computing environments where memory management is critical.
The solvers support any floating-point type, including extended and multi-precision types such as BigFloat, DoubleFloats or QuadMath.
Moreover, TRUNK, TRUNK-NLS, and FOMO support GPU arrays, broadening the range of hardware where the package can be effectively deployed, for instance when used together with `ExaModels.jl` [@shin2024accelerating].
The package documentation and \url{https://jso.dev/tutorials} provide examples illustrating the use of different floating-point systems.
Furthermore, the solvers expose in-place function variants, allowing multiple optimization problems with identical dimensions and data types to be solved efficiently without reallocations.

`JSOSolvers.jl` is built upon the JuliaSmoothOptimizers (JSO) tools [@The_JuliaSmoothOptimizers_Ecosystem].
JSO is an academic organization containing a collection of Julia packages for nonlinear optimization software development, testing, and benchmarking.
It provides tools for building models, accessing problem repositories, and solving subproblems.
Solvers in `JSOSolvers.jl` take as input an `AbstractNLPModel`, JSO's general model API defined in `NLPModels.jl` [@NLPModels_jl], a flexible data type to evaluate objective and constraints, their derivatives, and to provide any information that a solver might request from a model.
The user can hand-code derivatives, use automatic differentiation, or use JSO-interfaces to classical mathematical optimization modeling languages such as AMPL [@fourer2003ampl], CUTEst [@cutest], or JuMP [@jump]. 
The solvers rely heavily on iterative linear algebra methods from `Krylov.jl` [@Krylov_jl].

# Statement of need

Julia's JIT compiler is attractive for the design of efficient scientific computing software, and, in particular, mathematical optimization [@lubin2015computing], and has become a natural choice for developing new solvers.

While several options exist to solve \eqref{eq:nlp} in Julia, many rely on wrappers to solvers implemented in low-level compiled languages.
For example, if \eqref{eq:nlp} is modeled using JuMP [@jump], it can be passed to solvers like IPOPT [@wachter2006implementation] or Artelys Knitro [@byrd2006k] via Julia’s native C++ and Fortran interoperability.
However, these interfaces often lack flexibility with respect to data types and numerical precision.
In contrast, solvers written in pure Julia can operate seamlessly with a variety of arithmetic types or even GPU array types.
This capability is increasingly important as extended-precision arithmetic becomes more accessible through packages such as GNU MPFR, shipped with Julia.
Such flexibility enables high-precision computing when numerical accuracy is paramount.

Several alternatives to `JSOSolvers.jl` are available within and outside the Julia ecosystem.
`Optim.jl` [@mogensen2018optim] is a general-purpose optimization library in pure Julia, suitable for small to medium-scale problems, but it lacks in-place execution and GPU support.
`NLopt.jl` [@NLopt] provides access to a broad collection of optimization algorithms via a C library but does not support matrix-free methods or extended precision.
`AdaptiveRegularization.jl` [@AdaptiveRegularization_jl] offers a matrix-free, multi-precision solver for unconstrained problems and is closely aligned with the design philosophy of `JSOSolvers.jl`.
Ipopt [@wachter2006implementation], via `Ipopt.jl`, is widely used and efficient but requires explicit derivatives and is limited to CPU execution.
GALAHAD [@galahad], a Fortran-based suite for large-scale problems, is accessible through experimental Julia wrappers, yet lacks native composability.
Commercial solvers such as Artelys Knitro [@byrd2006k] provide robust algorithms but remain constrained by licensing and limited Julia interoperability.
`Optimization.jl` is a wrapper to existing optimization packages.

## Benchmarking

`JSOSolvers.jl` can solve large-scale problems and can be benchmarked easily against other JSO-compliant solvers using `SolverBenchmark.jl` [@SolverBenchmark_jl].
We include below performance profiles [@dolan2002benchmarking] with respect to elapsed time of `JSOSolvers.jl` solvers against Ipopt on all the 291 unconstrained problems from the CUTEst collection [@cutest], whose dimensions range from 2 up to 192,627 variables.

*LBFGS uses only first-order information, while TRON and TRUNK use Hessian-vector products and IPOPT uses the Hessian as a matrix.*

Without explaining performance profiles in full detail, the plot shows that Ipopt is fastest on 42 problems (15%), TRON on 9 (3%), TRUNK on 64 (21%), and L-BFGS on 176 (60%).
Nearly all problems were solved within the 20-minute limit: TRON solved 272 (93%), Ipopt 270, TRUNK 269, and L-BFGS 267.
Overall, these results are very encouraging.

<!--
```
include("make_problems_list.jl") # setup a file `list_problems.dat` with problem names
include("benchmark.jl") # run the benchmark and store the result in a JLD2 file
include("analyze_benchmark.jl") # make the figure
```
-->
![Unconstrained solvers on CUTEst with respect to the elapsed time.](2025-09-06_ipopt_lbfgs_trunk_tron_cutest_Float64_0_291_time_pp.pdf){ width=100% }

# Acknowledgements

Dominique Orban is partially supported by an NSERC Discovery Grant.

# References

