@testset "Solver" begin
  mutable struct OptNoSolver{T} <: AbstractOptSolver{T} end
  solver = OptNoSolver{Float64}()

  function solve!(::OptNoSolver{T}, nlp::AbstractNLPModel) where {T}
    return OptSolverOutput(
      :unknown,
      zeros(nlp.meta.nvar),
      nlp
    )
  end

  mutable struct DummyProblem <: AbstractNLPModel
    meta :: NLPModelMeta
    counters :: Counters
  end

  nlp = DummyProblem(NLPModelMeta(3), Counters())
  output = solve!(solver, nlp)
  @test output.status == :unknown
  @test output.solution == zeros(3)
  @test output.objective == Inf
end
