@testset "Solver" begin
  @testset "Basic tests" begin
    mutable struct OptNoSolver{T, S} <: AbstractOptSolver{T, S} end
    solver = OptNoSolver{Float64, Vector{Float64}}()

    function solve!(::OptNoSolver{T, S}, nlp::AbstractNLPModel) where {T, S}
      x = zeros(T, nlp.meta.nvar)
      return OptSolverOutput(:unknown, x, nlp, objective=obj(nlp, x))
    end

    nlp = ADNLPModel(x -> 1.0, zeros(3))
    output = solve!(solver, nlp)
    @test output.status == :unknown
    @test output.solution == zeros(3)
    @test output.objective == 1.0
    @test neval_obj(nlp) == 1
    reset_problem!(nlp)
    @test neval_obj(nlp) == 0
  end

  @testset "Multiprecision" begin
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2))
    solver = DummySolver(nlp)
    @test typeof(solver) == DummySolver{Float64, Vector{Float64}}
    output = with_logger(NullLogger()) do
      solve!(solver, nlp)
    end
    @test typeof(output) == OptSolverOutput{Float64, Vector{Float64}}
    x0 = BigFloat.(output.solution)

    solver = DummySolver(nlp, x0=x0)
    @test typeof(solver) == DummySolver{BigFloat, Vector{BigFloat}}
    @test_throws MethodError solve!(solver, nlp)
    output = with_logger(NullLogger()) do
      solve!(solver, nlp, x0=x0)
    end
    @test typeof(output) == OptSolverOutput{BigFloat, Vector{BigFloat}}
  end

  @testset "Grid search" begin
    problems = [
      ADNLPModel(x -> (x[1] - 1)^2 + b^2 * (x[2] - x[1]^2)^2, zeros(2)) for b = 2:10
    ]

    grid_output = grid_search_tune(DummySolver, problems)
    grid_output = grid_output[:]
    sort!(grid_output, by = x -> x[2])
    @test grid_output[1][2][2] == 0 # Solved all problem
    @test grid_output[end][2][2] == length(problems) # Solved no problems
  end
end
