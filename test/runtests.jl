# This package
using OptSolver

# Auxiliary packages
using ADNLPModels, NLPModels, SolverCore

# stdlib
using LinearAlgebra, Logging, Test

include("dummy_solver.jl")

include("solver.jl")
include("output.jl")
