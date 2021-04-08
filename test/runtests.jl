# This package
using OptSolver

# Auxiliary packages
using ADNLPModels, NLPModels, SolverCore

# stdlib
using LinearAlgebra, Logging, Test

include("dummy_solver.jl")

include("test_logging.jl")
include("test_stats.jl")