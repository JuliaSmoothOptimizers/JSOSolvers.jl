# JSOSolvers.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3991143.svg)](https://doi.org/10.5281/zenodo.3991143)
[![GitHub release](https://img.shields.io/github/release/JuliaSmoothOptimizers/JSOSolvers.jl.svg)](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://jso.dev/JSOSolvers.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://jso.dev/JSOSolvers.jl/latest)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/JSOSolvers.jl/branch/main/graph/badge.svg?token=eyiGsilbZx)](https://codecov.io/gh/JuliaSmoothOptimizers/JSOSolvers.jl)

![CI](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/JSOSolvers.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/JSOSolvers.jl)

This package provides optimization solvers curated by the JuliaSmoothOptimizers
organization for unconstrained optimization

    min f(x)

and bound-constrained optimization

    min f(x)     s.t.  ℓ ≤ x ≤ u

This package provides an implementation of four classic algorithms for unconstrained/bound-constrained nonlinear optimization:

- `lbfgs`: an implementation of a limited-memory BFGS line-search method for unconstrained minimization;
- `R2`: a first-order quadratic regularization method for unconstrained optimization;
- `tron`: a pure Julia implementation of TRON, a trust-region solver for bound-constrained optimization described in

    >  Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
    >  Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
    >  DOI: [10.1137/S1052623498345075](https://www.doi.org/10.1137/S1052623498345075)

    as well as a variant for nonlinear least-squares;
- `trunk`: a trust-region solver for unconstrained optimization using exact second derivatives. Our implementation follows the description given in

    >  A. R. Conn, N. I. M. Gould, and Ph. L. Toint,
    >  Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.
    >  SIAM, Philadelphia, USA, 2000.
    >  DOI: [10.1137/1.9780898719857](https://www.doi.org/10.1137/1.9780898719857)

    The package also contains a variant for nonlinear least-squares.

## Installation

`pkg> add JSOSolvers`

## Example

```julia
using JSOSolvers, ADNLPModels

# Rosenbrock
nlp = ADNLPModel(x -> 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2, [-1.2; 1.0])
stats = lbfgs(nlp) # or trunk, tron, R2
```

## How to cite

If you use JSOSolvers.jl in your work, please cite using the format given in [CITATION.cff](CITATION.cff).

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
