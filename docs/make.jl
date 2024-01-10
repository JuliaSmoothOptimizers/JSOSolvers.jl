using JSOSolvers
using Documenter

DocMeta.setdocmeta!(JSOSolvers, :DocTestSetup, :(using JSOSolvers); recursive = true)

makedocs(;
  modules = [JSOSolvers],
  doctest = true,
  linkcheck = true,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/blob/{commit}{path}#{line}",
  sitename = "JSOSolvers.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/JSOSolvers.jl",
    assets = ["assets/style.css"],
  ),
  pages = [
    "index.md",
    "solvers.md",
    "internal.md",
    "contributing.md",
    "developer.md",
    "reference.md",
  ],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/JSOSolvers.jl", push_preview = true)
