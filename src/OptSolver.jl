module OptSolver

# JSO
using NLPModels, SolverCore

include("problem.jl")
include("solver.jl")
include("output.jl")
include("traits.jl")

end
