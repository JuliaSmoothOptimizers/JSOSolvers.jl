var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Home-1",
    "page": "Home",
    "title": "JSOSolvers.jl documentation",
    "category": "section",
    "text": "This package provides optimization solvers curated by the JuliaSmoothOptimizers organization. All solvers are based on SolverTools.jl.TODO:[ ] Show benchmarks of these solvers (using SolverBenchmark.jl);\n[ ] Profile solver to identify bottlenecks;\n[ ] Use PkgBenchmark and SolverBenchmark to compare future PRs."
},

{
    "location": "#Solver-input-and-output-1",
    "page": "Home",
    "title": "Solver input and output",
    "category": "section",
    "text": "All solvers use the input and output given by SolverTools. Every solver has the following call signature:    stats = solver(nlp; x, atol, rtol, max_eval, max_time, ...)wherenlp is an AbstractNLPModel or some specialization, such as an AbstractNLSModel;\nx is the starting default (default: nlp.meta.x0);\natol is the absolute stopping tolerance (default: atol = √ϵ);\nrtol is the relative stopping tolerance (default: rtol = √ϵ);\nmax_eval is the maximum number of objective and constraints function evaluations (default: -1, which means no limit);\nmax_time is the maximum allowed elapsed time (default: 30.0);\nstats is a SolverTools.GenericExecutionStats with the output of the solver."
},

{
    "location": "solvers/#",
    "page": "Solvers",
    "title": "Solvers",
    "category": "page",
    "text": ""
},

{
    "location": "solvers/#Solvers-1",
    "page": "Solvers",
    "title": "Solvers",
    "category": "section",
    "text": "Solver listlbfgs\ntron\ntrunkProblem type Solvers\nUnconstrained NLP lbfgs, tron, trunk\nUnconstrained NLS trunk\nBound-constrained NLP tron"
},

{
    "location": "solvers/#JSOSolvers.lbfgs",
    "page": "Solvers",
    "title": "JSOSolvers.lbfgs",
    "category": "function",
    "text": "lbfgs(nlp)\n\nAn implementation of a limited memory BFGS line-search method for unconstrained minimization.\n\n\n\n\n\n"
},

{
    "location": "solvers/#JSOSolvers.tron",
    "page": "Solvers",
    "title": "JSOSolvers.tron",
    "category": "function",
    "text": "tron(nlp)\n\nA pure Julia implementation of a trust-region solver for bound-constrained optimization:\n\nmin f(x)    s.t.    ℓ ≦ x ≦ u\n\nTRON is described in\n\nChih-Jen Lin and Jorge J. Moré, Newton\'s Method for Large Bound-Constrained Optimization Problems, SIAM J. Optim., 9(4), 1100–1127, 1999.\n\n\n\n\n\n"
},

{
    "location": "solvers/#JSOSolvers.trunk",
    "page": "Solvers",
    "title": "JSOSolvers.trunk",
    "category": "function",
    "text": "trunk(nlp)\n\nA trust-region solver for unconstrained optimization using exact second derivatives.\n\nThis implementation follows the description given in [1]. The main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\n[1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint,     Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.     SIAM, Philadelphia, USA, 2000.     DOI: 10.1137/1.9780898719857.\n\n\n\n\n\ntrunk(nls)\n\nA trust-region solver for nonlinear least squares.\n\nThis implementation follows the description given in [1]. The main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\n[1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint,     Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.     SIAM, Philadelphia, USA, 2000.     DOI: 10.1137/1.9780898719857.\n\n\n\n\n\n"
},

{
    "location": "solvers/#Solver-list-1",
    "page": "Solvers",
    "title": "Solver list",
    "category": "section",
    "text": "lbfgs\ntron\ntrunk"
},

{
    "location": "internal/#",
    "page": "Internal",
    "title": "Internal",
    "category": "page",
    "text": ""
},

{
    "location": "internal/#JSOSolvers.projected_newton!",
    "page": "Internal",
    "title": "JSOSolvers.projected_newton!",
    "category": "function",
    "text": "projected_newton!(x, H, g, Δ, gctol, s, max_cgiter, ℓ, u)\n\nCompute an approximate solution d for\n\nmin q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ\n\nstarting from s.  The steps are computed using the conjugate gradient method projected on the active bounds.\n\n\n\n\n\n"
},

{
    "location": "internal/#JSOSolvers.projected_line_search!",
    "page": "Internal",
    "title": "JSOSolvers.projected_line_search!",
    "category": "function",
    "text": "s = projected_line_search!(x, H, g, d, ℓ, u; μ₀ = 1e-2)\n\nPerforms a projected line search, searching for a step size t such that\n\n0.5sᵀHs + sᵀg ≦ μ₀sᵀg,\n\nwhere s = P(x + t * d) - x, while remaining on the same face as x + d. Backtracking is performed from t = 1.0. x is updated in place.\n\n\n\n\n\n"
},

{
    "location": "internal/#JSOSolvers.cauchy",
    "page": "Internal",
    "title": "JSOSolvers.cauchy",
    "category": "function",
    "text": "α, s = cauchy(x, H, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)\n\nComputes a Cauchy step s = P(x - α g) - x for\n\nmin  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,\n\nwith the sufficient decrease condition\n\nq(s) ≦ μ₀sᵀg.\n\n\n\n\n\n"
},

{
    "location": "internal/#Internal-functions-1",
    "page": "Internal",
    "title": "Internal functions",
    "category": "section",
    "text": "JSOSolvers.projected_newton!\nJSOSolvers.projected_line_search!\nJSOSolvers.cauchy"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

]}
