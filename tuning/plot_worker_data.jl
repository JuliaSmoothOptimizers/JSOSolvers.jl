using JSON
ENV["GKSwstype"]="100"
using Plots
using Statistics
gr()

const IS_LOAD_BALANCING = false
const PATH_PREFIX = IS_LOAD_BALANCING ? "" : "no_"
const BASE_FILE_PATH = joinpath(@__DIR__, "plots", "$(PATH_PREFIX)load_balancing")

plot_data = JSON.parsefile(joinpath(BASE_FILE_PATH, "workers_times.json"), dicttype=Dict{String, Vector{Float64}})

nb_it = length(collect(values(plot_data))[1])
plot_data_transpose = [Float64[] for i in 1:nb_it]
for worker_times in values(plot_data)
  for it in 1:nb_it
    push!(plot_data_transpose[it], worker_times[it])
  end
end

nb_it = length(collect(values(plot_data))[1])
plot_data_transpose = [Float64[] for i in 1:nb_it]
for worker_times in values(plot_data)
  for it in 1:nb_it
    push!(plot_data_transpose[it], worker_times[it])
  end
end

# Total time for each bb eval 
let p = plot(2:nb_it+1, [sum(times) for times in plot_data_transpose], title="Total time for each black box evaluation", xlabel="black box iterations", ylabel="time (s)")
  savefig(p, joinpath(BASE_FILE_PATH, "total_times_per_iteration.png"))
end
# avg time with error bars
let p = plot(2:nb_it+1, [mean(times) for times in plot_data_transpose], ribbon=[std(times) for times in plot_data_transpose],fillalpha=.5, title="Mean time for each black box evaluation", xlabel="black box iterations", ylabel="time (s)")
  savefig(p, joinpath(BASE_FILE_PATH, "mean_times_per_iteration.png"))
end

# max worker time per eval
let p = plot(2:nb_it+1, [max(times...) for times in plot_data_transpose], title="max time for each black box evaluation", xlabel="black box iterations", ylabel="time (s)")
  savefig(p, joinpath(BASE_FILE_PATH, "max_times_per_iteration.png"))
end
# plotting all the workers together
let p = plot(1:1, 0, title="Workers time with load balancing", xlabel="Black box evaluations", ylabel="time (s)", lw=2)
  for (worker_id, y) in plot_data
    p = plot!(2:length(y)+1, y, label="Worker $worker_id")
  end 
  savefig(p, joinpath(BASE_FILE_PATH, "worker_times.png"))
end

# plotting each worker individually
for (worker_id, y) in plot_data
  local p = plot(2:length(y)+1, y, label = "Worker $worker_id", lw=2)
  savefig(p, joinpath(BASE_FILE_PATH, "worker_$worker_id.png"))
end

median_worker_times = Dict(w => median(worker_times) for (w, worker_times) in plot_data)
let p=scatter(2:2, 0, title="Median time of Workers with load balancing", xlabel="Workers", ylabel="median time (s)")
  for (worker_id, median_time) in median_worker_times
    p = scatter!([worker_id], [median_time], label="Worker $worker_id")
  end
  savefig(p, joinpath(BASE_FILE_PATH, "median_worker_times.png"))
end