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
function (::Type{Solver})(nlp::AbstractNLPModel; kwargs...) where {T, S, Solver <: AbstractOptSolver{T, S}}
  return Solver(nlp.meta; kwargs...)
end

function (::Type{Solver})(meta::AbstractNLPModelMeta; x0::S = meta.x0, kwargs...) where {Solver <: AbstractOptSolver, S}
  T = eltype(x0)
  return Solver{T, S}(meta; x0 = x0, kwargs...)
end

function (::Type{Solver})(nlp::AbstractNLPModel; x0::S = nlp.meta.x0, kwargs...) where {Solver <: AbstractOptSolver, S}
  T = eltype(x0)
  return Solver{T, S}(nlp.meta; x0 = x0, kwargs...)
end