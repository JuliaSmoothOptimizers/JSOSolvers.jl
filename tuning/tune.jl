using Pkg
using Distributed
using SolverParameters
using SolverTuning
using SolverCore
using NLPModels
using BenchmarkTools
using Random
using JSON

const IS_LOAD_BALANCING = false
const PATH_PREFIX = IS_LOAD_BALANCING ? "" : "no_"
const BASE_FILE_PATH = joinpath(@__DIR__, "plots", "$(PATH_PREFIX)load_balancing")
# 1. Launch workers
init_workers(;nb_nodes=20, exec_flags="--project=$(@__DIR__)")

# Contains worker data at each iteration
workers_data = Vector{Dict{Int, Float64}}()

# Contains nb of problems per worker at each iteration: 
nb_problems_per_worker = Vector{Dict{Int, Vector{String}}}()

# 2. make modules available to all workers:
@everywhere begin
  using JSOSolvers, 
  SolverTuning,
  OptimizationProblems,
  OptimizationProblems.ADNLPProblems,
  NLPModels,
  ADNLPModels
end


# 3. Setup problems
problems = (eval(p)(type=Val(Float64)) for p ∈ filter(x -> x != :ADNLPProblems && x != :scosine, names(OptimizationProblems.ADNLPProblems)))
problems = Iterators.filter(p -> unconstrained(p) &&  100 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)

Random.seed!(2017)
problem_dict = Dict(nlp => 30*rand(Float64) for nlp ∈ problems)

# 4. expose solver parameters
mem = AlgorithmicParameter(5, IntegerRange(1, 100), "mem")
τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")

lbfgs_params = [mem, τ₁, bk_max]

solver = LBFGSSolver(first(problems), lbfgs_params)

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
  bmark_results, stats_results, solver_results = eval_solver(lbfgs, args...;kwargs...)
  bmark_results = Dict(nlp => (median(bmark).time/1.0e9) for (nlp, bmark) ∈ bmark_results)
  total_time = sum(values(bmark_results))
  failure_penalty = count_failures(bmark_results, stats_results)
  bb_result = total_time + failure_penalty
  @info "failure_penalty: $failure_penalty"

  # Getting worker data:
  global workers_data
  global nb_problems_per_worker
  worker_times = Dict(worker_id => 0.0 for worker_id in keys(solver_results))
  for (worker_id, solver_result) in solver_results
    bmark_trials, _ = solver_result
    worker_times[worker_id] = sum(median(trial).time/1.0e9 for trial in values(bmark_trials))
  end
  push!(workers_data, worker_times)
  # Getting nb of problems per worker:
  problems_per_worker = Dict{Int, Vector{String}}()
  for (worker_id, solver_result) in solver_results
    bmark_trials, _ = solver_result
    problems_per_worker[worker_id] = [get_name(nlp) for nlp ∈ keys(bmark_trials)]
  end
  push!(nb_problems_per_worker, problems_per_worker)

  return [bb_result], bmark_results, stats_results
end
kwargs = Dict{Symbol, Any}(:verbose => 0, :max_time => 60.0)
black_box = BlackBox(solver, lbfgs_params, my_black_box, kwargs)        

# 6. define load balancer
lb = IS_LOAD_BALANCING ? GreedyLoadBalancer(problem_dict) : RoundRobinLoadBalancer(problem_dict)

# 7. define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box, lb)

# named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  # max_time = 300,
  max_bb_eval =200,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# 8. Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
@info ("Best feasible parameters: $(result.x_best_feas)")

# discard the first iteration
let plot_data = Dict{Int, Vector{Float64}}(worker_id => Float64[] for worker_id in workers())
  global workers_data
  for bb_iteration in workers_data[2:end]
    for (worker_id, time) ∈ bb_iteration
      push!(plot_data[worker_id], time)
    end
  end
  
  open(joinpath(BASE_FILE_PATH, "workers_times.json"), "w") do f
    JSON.print(f, plot_data)
  end
end

open(joinpath(BASE_FILE_PATH, "workers_problems.json"), "w") do f
  JSON.print(f, nb_problems_per_worker)
end

rmprocs(workers())
