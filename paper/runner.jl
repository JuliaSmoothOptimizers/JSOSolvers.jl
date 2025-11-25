using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
Pkg.instantiate()

include("make_problem_list.jl")
include("benchmark.jl")
include("analyze.jl")
