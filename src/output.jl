export OptSolverOutput

mutable struct OptSolverOutput{T} <: AbstractSolverOutput{T}
  status::Symbol
  solution
  objective::T # f(x)
  dual_feas::T # ‖∇f(x)‖₂ for unc, ‖P[x - ∇f(x)] - x‖₂ for bnd, etc.
  primal_feas::T # ‖c(x)‖ for equalities
  multipliers
  multipliers_L
  multipliers_U
  iter::Int
  counters::NLPModels.NLSCounters
  elapsed_time::Float64
  solver_specific::Dict{Symbol, Any}
end

function OptSolverOutput(
  status::Symbol,
  solution::AbstractArray{T},
  nlp::AbstractNLPModel;
  objective::T = T(Inf),
  dual_feas::T = T(Inf),
  primal_feas::T = unconstrained(nlp) || bound_constrained(nlp) ? zero(T) : T(Inf),
  multipliers::Vector = T[],
  multipliers_L::Vector = T[],
  multipliers_U::Vector = T[],
  iter::Int = -1,
  elapsed_time::Float64 = Inf,
  solver_specific::Dict = Dict{Symbol, Any}(),
) where {T}
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
  return OptSolverOutput{T}(
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
