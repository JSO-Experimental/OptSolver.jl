@testset "Solver" begin
  mutable struct OptNoSolver{T, S} <: AbstractOptSolver{T, S} end
  solver = OptNoSolver{Float64, Vector{Float64}}()

  function solve!(::OptNoSolver{T, S}, nlp::AbstractNLPModel) where {T, S}
    return OptSolverOutput(:unknown, zeros(T, nlp.meta.nvar), nlp)
  end

  mutable struct DummyProblem <: AbstractNLPModel
    meta::NLPModelMeta
    counters::Counters
  end

  nlp = DummyProblem(NLPModelMeta(3), Counters())
  output = solve!(solver, nlp)
  @test output.status == :unknown
  @test output.solution == zeros(3)
  @test output.objective == Inf
end
