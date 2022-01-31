using Pkg
using Distributed
using SolverParameters
using SolverTuning
using SolverCore
using NLPModels
using BenchmarkTools


# 1. Launch workers
init_workers(;nb_nodes=20, exec_flags="--project=$(@__DIR__)")

# 2. make modules available to all workers:
@everywhere begin
  using JSOSolvers, 
    SolverTuning,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    NLPModels,
    ADNLPModels
end
# @info "Main: $(names(Main))"
# @info "Main.JSOSolvers: $(names(Main.JSOSolvers))"
# @info "JSOSolvers: $(names(JSOSolvers))"

# 3. Setup problems
problems = (eval(p)(type=Val(Float64)) for p ∈ filter(x -> x != :ADNLPProblems && x != :scosine, names(OptimizationProblems.ADNLPProblems)))
problems = Iterators.filter(p -> unconstrained(p) &&  5 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems )
problem_dict = Dict(nlp => 30*rand(Float64) for nlp ∈ problems)

# 4. expose solver parameters
mem = AlgorithmicParameter(5, IntegerRange(1, 100), "mem")
τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")

lbfgs_params = [mem, τ₁, bk_max]

# This is ahack because AbstractOptSolver is in JSOSolvers...
solver = Main.LBFGSSolver(first(problems), lbfgs_params)

# Function that will count failures
function count_failures(bmark_results::Dict{P, Float64}, stats_results::Dict{AbstractNLPModel, AbstractExecutionStats}) where {P <: AbstractNLPModel}
  failure_penalty = 0.0   
  for (nlp, stats) in stats_results
    is_failure(stats) || continue
    failure_penalty += 25.0 * bmark_results[nlp]
  end
  return failure_penalty
  end

  function is_failure(stats::AbstractExecutionStats)
    failure_status = [:exception, :infeasible, :max_eval, :max_iter, :max_time, :stalled, :neg_pred]
    return any(s -> s == stats.status, failure_status)
  end

# 5. define user's blackbox:
function my_black_box(args...;kwargs...)
  bmark_results, stats_results = eval_solver(lbfgs, args...;kwargs...)
  bmark_results = Dict(nlp => (median(bmark).time/1.0e9) for (nlp, bmark) ∈ bmark_results)
  total_time = sum(values(bmark_results))
  failure_penalty = count_failures(bmark_results, stats_results)
  bb_result = total_time + failure_penalty
  @info "failure_penalty: $failure_penalty"

  return [bb_result], bmark_results, stats_results
end
kwargs = Dict{Symbol, Any}(:verbose => 0, :max_time => 60.0)
black_box = BlackBox(solver, lbfgs_params, my_black_box, kwargs)        

# 6. define load balancer
lb = GreedyLoadBalancer(problem_dict)

# 7. define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box, lb)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  max_time = 180,
  # max_bb_eval = 3,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# 8. Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
@info ("Best feasible parameters: $(result.x_best_feas)")