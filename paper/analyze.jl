using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
Pkg.instantiate()

using DataFrames, Dates, JLD2
using SolverBenchmark, SolverCore, NLPModels

name = "2025-09-06_ipopt_lbfgs_trunk_tron_cutest_Float64_0_291"
if name == "" # select the last jld2 file by default
  # TODO: this fails if there is a file with less than 4 characters
  files = filter(x -> x[(end - 4):end] == ".jld2", readdir(abspath(path)))
  if length(files) > 0
    name = files[end][1:(end - 5)]
  else
    @warn "A valid name should be provided for the JLD2 file."
  end
end
@load "$name.jld2" stats max_time tol

cols = [
  :id,
  :name,
  :nvar,
  :objective,
  :dual_feas,
  :neval_obj,
  :neval_grad,
  :neval_hess,
  :iter,
  :elapsed_time,
  :status,
]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
)

io = open(joinpath(path, "$name.dat"), "w")
for solver ∈ keys(stats)
  println(io, "Solver: $solver")
  pretty_stats(io, stats[solver][!, cols], hdr_override = header)
end
close(io)

# Path because of Ipopt on windows issue:
stats[:ipopt].elapsed_time .= stats[:ipopt].real_time

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time", "obj + grad + hess"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
]

nproblems = size(stats[:ipopt], 1)
@info "Number of problems solved"
for solver ∈ keys(stats)
  nsolved = sum(solved(stats[solver]))
  @info "$(solver): $nsolved ($(nsolved / nproblems * 100) %)"
end

tim = Dict()
for solver ∈ keys(stats)
  tim_solver = stats[solver][!, :elapsed_time]
  push!(tim, solver => tim_solver)
end
tim_best = ones(nproblems) * Inf # time of the best 
for i = 1:nproblems
  for solver ∈ keys(stats)
    tim_best[i] = min(tim[solver][i], tim_best[i])
  end
end
@info "Number of problems fastest"
for solver ∈ keys(stats)
  @info "$(solver): $(sum(tim[solver] .== tim_best)) ($(sum(tim[solver] .== tim_best) / nproblems * 100) %)"
end

using Plots
gr()

# Save individual performance profiles
for i = 1:length(costs)
  title_name = "" # "Unconstrained solvers on CUTEst w.r.t. $(costnames[i])"
  performance_profile(stats, costs[i], title = title_name)

  base = string(name, "_", costnames[i], "_pp")
  # Save as SVG and PDF (add "eps" here if you also want EPS)
  savefig("$base.svg")
  savefig("$base.pdf")
end

# Combined profile_solvers plot
profile_solvers(stats, costs, costnames)
# Save the combined figure as SVG and PDF
savefig("$name.svg")
savefig("$name.pdf")
