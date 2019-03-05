using Documenter, SolverTools

makedocs(
  modules = [SolverTools],
  doctest = true,
  linkcheck = true,
  strict = true,
  assets = ["assets/style.css"],
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "SolverTools.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Reference" => "reference.md",
          ]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/SolverTools.jl.git",
  target = "build",
  devbranch = "master"
)
