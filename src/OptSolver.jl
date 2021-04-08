module OptSolver

# stdlib
using Logging, Printf

# JSO
using NLPModels, SolverCore

include("logger.jl")
include("optsolver.jl")
include("optoutput.jl")
include("traits.jl")

end