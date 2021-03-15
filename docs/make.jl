using Documenter, SolverTools

makedocs(
  modules = [SolverTools],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "SolverTools.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Reference" => "reference.md",
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/SolverTools.jl.git",
  push_preview = true,
  devbranch = "master"
)
