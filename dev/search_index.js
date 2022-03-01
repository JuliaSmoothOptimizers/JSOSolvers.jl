var documenterSearchIndex = {"docs":
[{"location":"solvers/#Solvers","page":"Solvers","title":"Solvers","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Solver list","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"lbfgs\ntron\ntrunk","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Problem type Solvers\nUnconstrained NLP lbfgs, tron, trunk\nUnconstrained NLS trunk, tron\nBound-constrained NLP tron\nBound-constrained NLS tron","category":"page"},{"location":"solvers/#Solver-list","page":"Solvers","title":"Solver list","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"lbfgs\ntron\ntrunk","category":"page"},{"location":"solvers/#JSOSolvers.lbfgs","page":"Solvers","title":"JSOSolvers.lbfgs","text":"lbfgs(nlp; kwargs...)\n\nAn implementation of a limited memory BFGS line-search method for unconstrained minimization.\n\nFor advanced usage, first define a LBFGSSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = LBFGSSolver(nlp; mem::Int = 5)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nx::V = nlp.meta.x0: the initial guess.\nmem::Int = 5: memory parameter of the lbfgs algorithm.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nτ₁::T = T(0.9999): slope factor in the Wolfe condition when performing the line search.\nbk_max:: Int = 25: maximum number of backtracks when performing the line search.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nstats = lbfgs(nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nsolver = LBFGSSolver(nlp; mem = 5);\nstats = solve!(solver, nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\n\n\n\n\n\n\n","category":"function"},{"location":"solvers/#JSOSolvers.tron","page":"Solvers","title":"JSOSolvers.tron","text":"tron(nlp; kwargs...)\n\nA pure Julia implementation of a trust-region solver for bound-constrained optimization:\n\n    min f(x)    s.t.    ℓ ≦ x ≦ u\n\nFor advanced usage, first define a TronSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TronSolver(nlp)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\nμ₀::T = T(1e-2): algorithm parameter in (0, 0.5).\nμ₁::T = one(T): algorithm parameter in (0, +∞).\nσ::T = T(10): algorithm parameter in (1, +∞).\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nmax_cgiter::Int = 50: subproblem's iteration limit.\nuse_only_objgrad::Bool = false: If true, the algorithm uses only the function objgrad instead of obj and grad.\ncgtol::T = T(0.1): subproblem tolerance.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nmax_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)): maximum trust-region radius in the first-step chosen smaller than SolverTools.TRONTrustRegion max_radius.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nTRON is described in\n\nChih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained\nOptimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.\nDOI: 10.1137/S1052623498345075\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nstats = tron(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nsolver = TronSolver(nlp);\nstats = solve!(solver, nlp)\n\n\n\n\n\n\n\ntron(nls; kwargs...)\n\nA pure Julia implementation of a trust-region solver for bound-constrained nonlinear least-squares problems:\n\nmin ½‖F(x)‖²    s.t.    ℓ ≦ x ≦ u\n\nArguments\n\nnls::AbstractNLSModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\nsubsolver::Symbol = :lsmr: Krylov.jl method used as subproblem solver, see JSOSolvers.tronls_allowed_subsolvers for a list.\nμ₀::T = T(1e-2): algorithm parameter in (0, 0.5).\nμ₁::T = one(T): algorithm parameter in (0, +∞).\nσ::T = T(10): algorithm parameter in (1, +∞).\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nmax_cgiter::Int = 50: subproblem iteration limit.\ncgtol::T = T(0.1): subproblem tolerance.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis is an adaptation for bound-constrained nonlinear least-squares problems of the TRON method described in\n\nChih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained\nOptimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.\nDOI: 10.1137/S1052623498345075\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))\nstats = tron(nls)\n\n\n\n\n\n","category":"function"},{"location":"solvers/#JSOSolvers.trunk","page":"Solvers","title":"JSOSolvers.trunk","text":"trunk(nlp; kwargs...)\n\nA trust-region solver for unconstrained optimization using exact second derivatives.\n\nFor advanced usage, first define a TrunkSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TrunkSolver(nlp, subsolver_type::Type{<:KrylovSolver} = CgSolver)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nbk_max::Int = 10: algorithm parameter.\nmonotone::Bool = true: algorithm parameter.\nnm_itmax::Int = 25: algorithm parameter.\nverbose::Int = 0: if > 0, display iteration information every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis implementation follows the description given in\n\nA. R. Conn, N. I. M. Gould, and Ph. L. Toint,\nTrust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.\nSIAM, Philadelphia, USA, 2000.\nDOI: 10.1137/1.9780898719857\n\nThe main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nstats = trunk(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nsolver = TrunkSolver(nlp)\nstats = solve!(solver, nlp)\n\n\n\n\n\n\n\ntrunk(nls; kwargs...)\n\nA pure Julia implementation of a trust-region solver for nonlinear least-squares problems:\n\nmin ½‖F(x)‖²\n\nArguments\n\nnls::AbstractNLSModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nx::V = nlp.meta.x0: the initial guess.\nsubsolver::Symbol = :lsmr: Krylov.jl method used as subproblem solver, see JSOSolvers.trunkls_allowed_subsolvers for a list.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nbk_max::Int = 10: algorithm parameter.\nmonotone::Bool = true: algorithm parameter.\nnm_itmax::Int = 25: algorithm parameter.\ntrsolver_args::Dict{Symbol, Any} = Dict{Symbol, Any}(): additional keyword arguments for the subproblem solver.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis implementation follows the description given in\n\nA. R. Conn, N. I. M. Gould, and Ph. L. Toint,\nTrust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.\nSIAM, Philadelphia, USA, 2000.\nDOI: 10.1137/1.9780898719857\n\nThe main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2)\nstats = trunk(nls)\n\n\n\n\n\n","category":"function"},{"location":"internal/#Internal-functions","page":"Internal","title":"Internal functions","text":"","category":"section"},{"location":"internal/","page":"Internal","title":"Internal","text":"JSOSolvers.projected_newton!\nJSOSolvers.projected_line_search!\nJSOSolvers.cauchy\nJSOSolvers.compute_As_slope_qs!\nJSOSolvers.projected_gauss_newton!\nJSOSolvers.projected_line_search_ls!\nJSOSolvers.cauchy_ls","category":"page"},{"location":"internal/#JSOSolvers.projected_newton!","page":"Internal","title":"JSOSolvers.projected_newton!","text":"projected_newton!(x, H, g, Δ, gctol, s, max_cgiter, ℓ, u)\n\nCompute an approximate solution d for\n\nmin q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ\n\nstarting from s.  The steps are computed using the conjugate gradient method projected on the active bounds.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.projected_line_search!","page":"Internal","title":"JSOSolvers.projected_line_search!","text":"s = projected_line_search!(x, H, g, d, ℓ, u; μ₀ = 1e-2)\n\nPerforms a projected line search, searching for a step size t such that\n\n0.5sᵀHs + sᵀg ≦ μ₀sᵀg,\n\nwhere s = P(x + t * d) - x, while remaining on the same face as x + d. Backtracking is performed from t = 1.0. x is updated in place.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.cauchy","page":"Internal","title":"JSOSolvers.cauchy","text":"α, s = cauchy(x, H, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)\n\nComputes a Cauchy step s = P(x - α g) - x for\n\nmin  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,\n\nwith the sufficient decrease condition\n\nq(s) ≦ μ₀sᵀg.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.compute_As_slope_qs!","page":"Internal","title":"JSOSolvers.compute_As_slope_qs!","text":"slope, qs = compute_As_slope_qs!(As, A, s, Fx)\n\nCompute slope = dot(As, Fx) and qs = dot(As, As) / 2 + slope. Use As to store A * s.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.projected_gauss_newton!","page":"Internal","title":"JSOSolvers.projected_gauss_newton!","text":"projected_gauss_newton!(x, A, Fx, Δ, gctol, s, max_cgiter, ℓ, u)\n\nCompute an approximate solution d for\n\nmin q(d) = ½‖Ad + Fx‖² - ½‖Fx‖²     s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ\n\nstarting from s.  The steps are computed using the conjugate gradient method projected on the active bounds.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.projected_line_search_ls!","page":"Internal","title":"JSOSolvers.projected_line_search_ls!","text":"s = projected_line_search_ls!(x, A, g, d, ℓ, u; μ₀ = 1e-2)\n\nPerforms a projected line search, searching for a step size t such that\n\n½‖As + Fx‖² ≤ ½‖Fx‖² + μ₀FxᵀAs\n\nwhere s = P(x + t * d) - x, while remaining on the same face as x + d. Backtracking is performed from t = 1.0. x is updated in place.\n\n\n\n\n\n","category":"function"},{"location":"internal/#JSOSolvers.cauchy_ls","page":"Internal","title":"JSOSolvers.cauchy_ls","text":"α, s = cauchy_ls(x, A, Fx, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)\n\nComputes a Cauchy step s = P(x - α g) - x for\n\nmin  q(s) = ½‖As + Fx‖² - ½‖Fx‖²     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,\n\nwith the sufficient decrease condition\n\nq(s) ≦ μ₀gᵀs,\n\nwhere g = AᵀFx.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [JSOSolvers]","category":"page"},{"location":"reference/#JSOSolvers.LBFGSSolver","page":"Reference","title":"JSOSolvers.LBFGSSolver","text":"lbfgs(nlp; kwargs...)\n\nAn implementation of a limited memory BFGS line-search method for unconstrained minimization.\n\nFor advanced usage, first define a LBFGSSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = LBFGSSolver(nlp; mem::Int = 5)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nx::V = nlp.meta.x0: the initial guess.\nmem::Int = 5: memory parameter of the lbfgs algorithm.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nτ₁::T = T(0.9999): slope factor in the Wolfe condition when performing the line search.\nbk_max:: Int = 25: maximum number of backtracks when performing the line search.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nstats = lbfgs(nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nsolver = LBFGSSolver(nlp; mem = 5);\nstats = solve!(solver, nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\n\n\n\n\n","category":"type"},{"location":"reference/#JSOSolvers.TronSolver","page":"Reference","title":"JSOSolvers.TronSolver","text":"tron(nlp; kwargs...)\n\nA pure Julia implementation of a trust-region solver for bound-constrained optimization:\n\n    min f(x)    s.t.    ℓ ≦ x ≦ u\n\nFor advanced usage, first define a TronSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TronSolver(nlp)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\nμ₀::T = T(1e-2): algorithm parameter in (0, 0.5).\nμ₁::T = one(T): algorithm parameter in (0, +∞).\nσ::T = T(10): algorithm parameter in (1, +∞).\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nmax_cgiter::Int = 50: subproblem's iteration limit.\nuse_only_objgrad::Bool = false: If true, the algorithm uses only the function objgrad instead of obj and grad.\ncgtol::T = T(0.1): subproblem tolerance.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nmax_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)): maximum trust-region radius in the first-step chosen smaller than SolverTools.TRONTrustRegion max_radius.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nTRON is described in\n\nChih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained\nOptimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.\nDOI: 10.1137/S1052623498345075\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nstats = tron(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nsolver = TronSolver(nlp);\nstats = solve!(solver, nlp)\n\n\n\n\n\n","category":"type"},{"location":"reference/#JSOSolvers.TrunkSolver","page":"Reference","title":"JSOSolvers.TrunkSolver","text":"trunk(nlp; kwargs...)\n\nA trust-region solver for unconstrained optimization using exact second derivatives.\n\nFor advanced usage, first define a TrunkSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TrunkSolver(nlp, subsolver_type::Type{<:KrylovSolver} = CgSolver)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nbk_max::Int = 10: algorithm parameter.\nmonotone::Bool = true: algorithm parameter.\nnm_itmax::Int = 25: algorithm parameter.\nverbose::Int = 0: if > 0, display iteration information every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis implementation follows the description given in\n\nA. R. Conn, N. I. M. Gould, and Ph. L. Toint,\nTrust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.\nSIAM, Philadelphia, USA, 2000.\nDOI: 10.1137/1.9780898719857\n\nThe main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nstats = trunk(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nsolver = TrunkSolver(nlp)\nstats = solve!(solver, nlp)\n\n\n\n\n\n","category":"type"},{"location":"reference/#JSOSolvers.cauchy-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, Real, Real, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.cauchy","text":"α, s = cauchy(x, H, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)\n\nComputes a Cauchy step s = P(x - α g) - x for\n\nmin  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,\n\nwith the sufficient decrease condition\n\nq(s) ≦ μ₀sᵀg.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.cauchy_ls-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, AbstractVector{T}, Real, Real, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.cauchy_ls","text":"α, s = cauchy_ls(x, A, Fx, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)\n\nComputes a Cauchy step s = P(x - α g) - x for\n\nmin  q(s) = ½‖As + Fx‖² - ½‖Fx‖²     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,\n\nwith the sufficient decrease condition\n\nq(s) ≦ μ₀gᵀs,\n\nwhere g = AᵀFx.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.compute_As_slope_qs!-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.compute_As_slope_qs!","text":"slope, qs = compute_As_slope_qs!(As, A, s, Fx)\n\nCompute slope = dot(As, Fx) and qs = dot(As, As) / 2 + slope. Use As to store A * s.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.lbfgs-Union{Tuple{NLPModels.AbstractNLPModel}, Tuple{V}} where V","page":"Reference","title":"JSOSolvers.lbfgs","text":"lbfgs(nlp; kwargs...)\n\nAn implementation of a limited memory BFGS line-search method for unconstrained minimization.\n\nFor advanced usage, first define a LBFGSSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = LBFGSSolver(nlp; mem::Int = 5)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nx::V = nlp.meta.x0: the initial guess.\nmem::Int = 5: memory parameter of the lbfgs algorithm.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nτ₁::T = T(0.9999): slope factor in the Wolfe condition when performing the line search.\nbk_max:: Int = 25: maximum number of backtracks when performing the line search.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nstats = lbfgs(nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3));\nsolver = LBFGSSolver(nlp; mem = 5);\nstats = solve!(solver, nlp)\n\n# output\n\n\"Execution stats: first-order stationary\"\n\n\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.projected_gauss_newton!-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, Real, Real, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.projected_gauss_newton!","text":"projected_gauss_newton!(x, A, Fx, Δ, gctol, s, max_cgiter, ℓ, u)\n\nCompute an approximate solution d for\n\nmin q(d) = ½‖Ad + Fx‖² - ½‖Fx‖²     s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ\n\nstarting from s.  The steps are computed using the conjugate gradient method projected on the active bounds.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.projected_line_search!-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.projected_line_search!","text":"s = projected_line_search!(x, H, g, d, ℓ, u; μ₀ = 1e-2)\n\nPerforms a projected line search, searching for a step size t such that\n\n0.5sᵀHs + sᵀg ≦ μ₀sᵀg,\n\nwhere s = P(x + t * d) - x, while remaining on the same face as x + d. Backtracking is performed from t = 1.0. x is updated in place.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.projected_line_search_ls!-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.projected_line_search_ls!","text":"s = projected_line_search_ls!(x, A, g, d, ℓ, u; μ₀ = 1e-2)\n\nPerforms a projected line search, searching for a step size t such that\n\n½‖As + Fx‖² ≤ ½‖Fx‖² + μ₀FxᵀAs\n\nwhere s = P(x + t * d) - x, while remaining on the same face as x + d. Backtracking is performed from t = 1.0. x is updated in place.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.projected_newton!-Union{Tuple{T}, Tuple{AbstractVector{T}, Union{AbstractMatrix, LinearOperators.AbstractLinearOperator}, AbstractVector{T}, Real, Real, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}} where T<:Real","page":"Reference","title":"JSOSolvers.projected_newton!","text":"projected_newton!(x, H, g, Δ, gctol, s, max_cgiter, ℓ, u)\n\nCompute an approximate solution d for\n\nmin q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ\n\nstarting from s.  The steps are computed using the conjugate gradient method projected on the active bounds.\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.tron-Tuple{Val{:GaussNewton}, NLPModels.AbstractNLSModel}","page":"Reference","title":"JSOSolvers.tron","text":"tron(nls; kwargs...)\n\nA pure Julia implementation of a trust-region solver for bound-constrained nonlinear least-squares problems:\n\nmin ½‖F(x)‖²    s.t.    ℓ ≦ x ≦ u\n\nArguments\n\nnls::AbstractNLSModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\nsubsolver::Symbol = :lsmr: Krylov.jl method used as subproblem solver, see JSOSolvers.tronls_allowed_subsolvers for a list.\nμ₀::T = T(1e-2): algorithm parameter in (0, 0.5).\nμ₁::T = one(T): algorithm parameter in (0, +∞).\nσ::T = T(10): algorithm parameter in (1, +∞).\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nmax_cgiter::Int = 50: subproblem iteration limit.\ncgtol::T = T(0.1): subproblem tolerance.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis is an adaptation for bound-constrained nonlinear least-squares problems of the TRON method described in\n\nChih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained\nOptimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.\nDOI: 10.1137/S1052623498345075\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))\nstats = tron(nls)\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.tron-Union{Tuple{V}, Tuple{Val{:Newton}, NLPModels.AbstractNLPModel}} where V","page":"Reference","title":"JSOSolvers.tron","text":"tron(nlp; kwargs...)\n\nA pure Julia implementation of a trust-region solver for bound-constrained optimization:\n\n    min f(x)    s.t.    ℓ ≦ x ≦ u\n\nFor advanced usage, first define a TronSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TronSolver(nlp)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\nμ₀::T = T(1e-2): algorithm parameter in (0, 0.5).\nμ₁::T = one(T): algorithm parameter in (0, +∞).\nσ::T = T(10): algorithm parameter in (1, +∞).\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nmax_cgiter::Int = 50: subproblem's iteration limit.\nuse_only_objgrad::Bool = false: If true, the algorithm uses only the function objgrad instead of obj and grad.\ncgtol::T = T(0.1): subproblem tolerance.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\nmax_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)): maximum trust-region radius in the first-step chosen smaller than SolverTools.TRONTrustRegion max_radius.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nTRON is described in\n\nChih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained\nOptimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.\nDOI: 10.1137/S1052623498345075\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nstats = tron(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x), ones(3), zeros(3), 2 * ones(3));\nsolver = TronSolver(nlp);\nstats = solve!(solver, nlp)\n\n\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.trunk-Tuple{Val{:GaussNewton}, NLPModels.AbstractNLSModel}","page":"Reference","title":"JSOSolvers.trunk","text":"trunk(nls; kwargs...)\n\nA pure Julia implementation of a trust-region solver for nonlinear least-squares problems:\n\nmin ½‖F(x)‖²\n\nArguments\n\nnls::AbstractNLSModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nx::V = nlp.meta.x0: the initial guess.\nsubsolver::Symbol = :lsmr: Krylov.jl method used as subproblem solver, see JSOSolvers.trunkls_allowed_subsolvers for a list.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nbk_max::Int = 10: algorithm parameter.\nmonotone::Bool = true: algorithm parameter.\nnm_itmax::Int = 25: algorithm parameter.\ntrsolver_args::Dict{Symbol, Any} = Dict{Symbol, Any}(): additional keyword arguments for the subproblem solver.\nverbose::Int = 0: if > 0, display iteration details every verbose iteration.\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis implementation follows the description given in\n\nA. R. Conn, N. I. M. Gould, and Ph. L. Toint,\nTrust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.\nSIAM, Philadelphia, USA, 2000.\nDOI: 10.1137/1.9780898719857\n\nThe main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2)\nstats = trunk(nls)\n\n\n\n\n\n","category":"method"},{"location":"reference/#JSOSolvers.trunk-Union{Tuple{V}, Tuple{Val{:Newton}, NLPModels.AbstractNLPModel}} where V","page":"Reference","title":"JSOSolvers.trunk","text":"trunk(nlp; kwargs...)\n\nA trust-region solver for unconstrained optimization using exact second derivatives.\n\nFor advanced usage, first define a TrunkSolver to preallocate the memory used in the algorithm, and then call solve!.\n\nsolver = TrunkSolver(nlp, subsolver_type::Type{<:KrylovSolver} = CgSolver)\nsolve!(solver, nlp; kwargs...)\n\nArguments\n\nnlp::AbstractNLPModel{T, V} represents the model to solve, see NLPModels.jl.\n\nThe keyword arguments may include\n\nsubsolver_logger::AbstractLogger = NullLogger(): subproblem's logger.\nx::V = nlp.meta.x0: the initial guess.\natol::T = √eps(T): absolute tolerance.\nrtol::T = √eps(T): relative tolerance, the algorithm stops when ||∇f(xᵏ)|| ≤ atol + rtol * ||∇f(x⁰)||.\nmax_eval::Int = -1: maximum number of objective function evaluations.\nmax_time::Float64 = 30.0: maximum time limit in seconds.\nbk_max::Int = 10: algorithm parameter.\nmonotone::Bool = true: algorithm parameter.\nnm_itmax::Int = 25: algorithm parameter.\nverbose::Int = 0: if > 0, display iteration information every verbose iteration.\nverbose_subsolver::Int = 0: if > 0, display iteration information every verbose_subsolver iteration of the subsolver.\n\nOutput\n\nThe returned value is a GenericExecutionStats, see SolverCore.jl.\n\nReferences\n\nThis implementation follows the description given in\n\nA. R. Conn, N. I. M. Gould, and Ph. L. Toint,\nTrust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.\nSIAM, Philadelphia, USA, 2000.\nDOI: 10.1137/1.9780898719857\n\nThe main algorithm follows the basic trust-region method described in Section 6. The backtracking linesearch follows Section 10.3.2. The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.\n\nExamples\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nstats = trunk(nlp)\n\nusing JSOSolvers, ADNLPModels\nnlp = ADNLPModel(x -> sum(x.^2), ones(3))\nsolver = TrunkSolver(nlp)\nstats = solve!(solver, nlp)\n\n\n\n\n\n\n\n","category":"method"},{"location":"reference/#Krylov.solve!","page":"Reference","title":"Krylov.solve!","text":"solve!(solver, nlp)\n\nSolve nlp using solver.\n\n\n\n\n\n","category":"function"},{"location":"#Home","page":"Home","title":"JSOSolvers.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides optimization solvers curated by the JuliaSmoothOptimizers organization. All solvers are based on SolverTools.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"TODO:","category":"page"},{"location":"","page":"Home","title":"Home","text":"[ ] Show benchmarks of these solvers (using SolverBenchmark.jl);\n[ ] Profile solver to identify bottlenecks;\n[ ] Use PkgBenchmark and SolverBenchmark to compare future PRs.","category":"page"},{"location":"#Solver-input-and-output","page":"Home","title":"Solver input and output","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"All solvers use the input and output given by SolverTools. Every solver has the following call signature:","category":"page"},{"location":"","page":"Home","title":"Home","text":"    stats = solver(nlp; x, atol, rtol, max_eval, max_time, ...)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where","category":"page"},{"location":"","page":"Home","title":"Home","text":"nlp is an AbstractNLPModel or some specialization, such as an AbstractNLSModel;\nx is the starting default (default: nlp.meta.x0);\natol is the absolute stopping tolerance (default: atol = √ϵ);\nrtol is the relative stopping tolerance (default: rtol = √ϵ);\nmax_eval is the maximum number of objective and constraints function evaluations (default: -1, which means no limit);\nmax_time is the maximum allowed elapsed time (default: 30.0);\nstats is a SolverTools.GenericExecutionStats with the output of the solver.","category":"page"}]
}
