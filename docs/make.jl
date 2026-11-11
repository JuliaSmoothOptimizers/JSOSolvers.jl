using Documenter, JSOSolvers

makedocs(
  modules = [JSOSolvers],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
    size_threshold = 400 * 2^10, # 400 KiB
  ),
  sitename = "JSOSolvers.jl",
  pages = [
    "index.md",
    "examples.md",
    "solvers.md",
    "benchmark.md",
    "floating-point-systems.md",
    "performance-tips.md",
    "internal.md",
    "reference.md",
  ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/JSOSolvers.jl.git",
  push_preview = true,
  devbranch = "main",
)
