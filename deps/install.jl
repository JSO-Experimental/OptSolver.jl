Pkg.add("Compat")
using Compat
import Compat.String

const home = "https://github.com/JuliaSmoothOptimizers"

const deps = Dict{String, String}(
              "OptimizationProblems" => "master",
              "Krylov" => "develop",
              "BenchmarkProfiles" => "master")

function dep_installed(dep)
  try
    # throws an error or returns nothing
    # (https://github.com/JuliaLang/julia/issues/16300)
    return Pkg.installed(dep) != nothing
  catch
    return false
  end
end

function dep_install(dep, branch)
  dep_installed(dep) || Pkg.clone("$home/$dep.jl.git")
  Pkg.checkout(dep, branch)
  Pkg.build(dep)
end

function deps_install(deps)
  for dep in keys(deps)
    dep_install(dep, deps[dep])
  end
end

deps_install(deps)
