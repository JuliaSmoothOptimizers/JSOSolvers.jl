using Documenter, JSOSolvers

makedocs(
  modules = [JSOSolvers],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
  ),
  sitename = "JSOSolvers.jl",
  pages = ["index.md", "solvers.md", "internal.md", "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/JSOSolvers.jl.git",
  push_preview = true,
  devbranch = "main",
)
