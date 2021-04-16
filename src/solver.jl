export AbstractOptSolver

"""
    AbstractOptSolver{T, S} <: AbstractSolver{T, S}

Base type of JSO-compliant optimization solvers.
Like `AbstractSolver{T, S}`, a solver must have three member:
- `initialized :: Bool`, indicating whether the solver was initialized
- `params :: Dict`, a dictionary of parameters for the solver
- `workspace`, a named tuple with arrays used by the solver.

In addition, a `Solver{T, S} <: AbstractOptSolver{T, S}` must define the `solve!` function
"""
abstract type AbstractOptSolver{T, S} <: AbstractSolver{T, S} end

#=
Expected constructos: Solver(meta)
=#
(::Type{S})(nlp::AbstractNLPModel; kwargs...) where {S <: AbstractOptSolver} =
  S(nlp.meta; kwargs...)
