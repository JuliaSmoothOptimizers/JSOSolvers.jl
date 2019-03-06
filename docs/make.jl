using Documenter, JSOSolvers

makedocs(
  modules = [JSOSolvers],
  doctest = true,
  linkcheck = true,
  strict = true,
  assets = ["assets/style.css"],
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "JSOSolvers.jl",
  pages = ["Home" => "index.md",
           "Solvers" => "solvers.md",
           "Internal" => "internal.md",
           "Reference" => "reference.md",
          ]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/JSOSolvers.jl.git",
  target = "build",
  devbranch = "master"
)
