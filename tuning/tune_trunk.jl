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

T = Float64
# 3. Setup problems
problems = (eval(p)(type=Val(T)) for p ∈ filter(x -> x != :ADNLPProblems && x != :scosine, names(OptimizationProblems.ADNLPProblems)))
problems = Iterators.filter(p -> unconstrained(p) &&  100 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)


# Function that will count failures
function count_failures(bmark_results::Dict{Int, Float64}, stats_results::Dict{Int, AbstractExecutionStats})
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
  bmark_results, stats_results, solver_results = eval_solver(trunk, args...;kwargs...)
  bmark_results_output = Dict(pb_id => (median(bmark).time/1.0e9 * median(bmark).memory/1.0e6) for (pb_id, bmark) ∈ bmark_results)
  
  total_output = sum(values(bmark_results_output))
  failure_penalty = count_failures(bmark_results_output, stats_results)
  bb_result = total_output + failure_penalty
  @info "failure_penalty: $failure_penalty"

  return [bb_result], bmark_results_output, stats_results
end
solver = TrunkSolver(first(problems))
kwargs = Dict{Symbol, Any}(:verbose => 0, :max_time => 60.0)
black_box = BlackBox(solver, collect(values(solver.parameters)), my_black_box, kwargs)        

# 7. define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box, problems)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  # max_time = 300,
  max_bb_eval =300,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# 8. Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
@info ("Best feasible parameters: $(result.x_best_feas)")

for p in black_box.solver_params
    @info "$(name(p)): $(default(p))"
end
rmprocs(workers())
