using Documenter, JSOSolvers

makedocs(
  modules = [JSOSolvers],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
  ),
  sitename = "JSOSolvers.jl",
  pages = [
    "Home" => "index.md",
    "Solvers" => "solvers.md",
    "Internal" => "internal.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/JSOSolvers.jl.git", devbranch = "main")
