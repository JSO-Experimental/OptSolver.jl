@testset "Solver" begin
  @testset "Basic tests" begin
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

  @testset "Multiprecision" begin
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2))
    solver = DummySolver(nlp)
    @test typeof(solver) == DummySolver{Float64, Vector{Float64}}
    output = solve!(solver, nlp)
    @test typeof(output) == OptSolverOutput{Float64, Vector{Float64}}
    x0 = BigFloat.(output.solution)

    solver = DummySolver(nlp, x0=x0)
    @test typeof(solver) == DummySolver{BigFloat, Vector{BigFloat}}
    @test_throws MethodError solve!(solver, nlp)
    output = solve!(solver, nlp, x0=x0)
    @test typeof(output) == OptSolverOutput{BigFloat, Vector{BigFloat}}
  end
end
