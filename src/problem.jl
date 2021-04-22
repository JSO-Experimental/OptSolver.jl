SolverCore.reset_problem!(nlp::AbstractNLPModel) = reset!(nlp)

SolverCore.problem_name(nlp::AbstractNLPModel) = nlp.meta.name

SolverCore.problem_info(::Type{<: AbstractNLPModel}) = (
  [Int, Int],
  [:nvar, :ncon],
  nlp -> [nlp.meta.nvar, nlp.meta.ncon]
)

SolverCore.problem_info(::Type{<: AbstractNLSModel}) = (
  [Int, Int, Int],
  [:nvar, :ncon, :nequ],
  nls -> [nls.meta.nvar, nls.meta.ncon, nls.nls_meta.nequ]
)

function SolverCore.output_info(::Type{<: AbstractNLPModel})
  nlp_counters = fieldnames(Counters) |> collect
  nls_counters = setdiff(fieldnames(NLSCounters), [:counters])
  types = fill(Int, length(nlp_counters) + length(nls_counters))
  names = [nlp_counters; nls_counters]
  foo = o -> [
    [getfield(o.counters.counters, f) for f in nlp_counters];
    [getfield(o.counters, f) for f in nls_counters]
  ]
  return types, names, foo
end