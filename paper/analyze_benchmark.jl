using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)

using DataFrames, Dates, JLD2
using SolverBenchmark, SolverCore, NLPModels

name = "2024-09-25_ipopt_lbfgs_trunk_tron_adop_Float64_372"
if name == "" # select the last jld2 file by default
  # TODO: this fails if there is a file with less than 4 characters
  files = filter(x -> x[(end - 4):end] == ".jld2", readdir(abspath(path)))
  if length(files) > 0
    name = files[end - 5:end]
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
  pretty_stats(io, stats[solver][!, cols], hdr_override = header)
end

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time", "obj + grad + hess"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
]

using Plots
gr()

profile_solvers(stats, costs, costnames)
png("$name")
