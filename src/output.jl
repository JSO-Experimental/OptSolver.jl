export OptSolverOutput

SolverCore.solver_output_type(::Type{<: AbstractOptSolver{T, S}}) where {T, S} = OptSolverOutput{T, S}
SolverCore.solver_output_type(::Type{<: AbstractOptSolver}) = OptSolverOutput{Number, Any}

mutable struct OptSolverOutput{T, S} <: AbstractSolverOutput{T, S}
  status::Symbol
  solution::S
  objective::T # f(x)
  dual_feas::T # ‖∇f(x)‖₂ for unc, ‖P[x - ∇f(x)] - x‖₂ for bnd, etc.
  primal_feas::T # ‖c(x)‖ for equalities
  multipliers::S
  multipliers_L::S
  multipliers_U::S
  iter::Int
  counters::NLPModels.NLSCounters
  elapsed_time::Float64
  solver_specific::Dict{Symbol, Any}
end

function OptSolverOutput(status::Symbol, solution::S, nlp::AbstractNLPModel; kwargs...) where {S}
  OptSolverOutput{eltype(solution)}(status, solution, nlp; kwargs...)
end

function OptSolverOutput{T}(
  status::Symbol,
  solution::S,
  nlp::AbstractNLPModel;
  objective::T = T(Inf),
  dual_feas::T = T(Inf),
  primal_feas::T = unconstrained(nlp) || bound_constrained(nlp) ? zero(T) : T(Inf),
  multipliers::Vector = S(undef, 0),
  multipliers_L::Vector = S(undef, 0),
  multipliers_U::Vector = S(undef, 0),
  iter::Int = -1,
  elapsed_time::Float64 = Inf,
  solver_specific::Dict = Dict{Symbol, Any}(),
) where {T, S}
  if !(status in keys(SolverCore.STATUSES))
    @error "status $status is not a valid status. Use one of the following: " join(
      keys(STATUSES),
      ", ",
    )
    throw(KeyError(status))
  end
  c = NLSCounters()
  for counter in fieldnames(Counters)
    setfield!(c.counters, counter, eval(Meta.parse("$counter"))(nlp))
  end
  if nlp isa AbstractNLSModel
    for counter in fieldnames(NLSCounters)
      counter == :counters && continue
      setfield!(c, counter, eval(Meta.parse("$counter"))(nlp))
    end
  end
  return OptSolverOutput{T, S}(
    status,
    solution,
    objective,
    dual_feas,
    primal_feas,
    multipliers,
    multipliers_L,
    multipliers_U,
    iter,
    c,
    elapsed_time,
    solver_specific,
  )
end
