@testset "OptOutput" begin
  nlp = ADNLPModel(x -> dot(x, x), zeros(2))
  output = OptSolverOutput(
    :first_order,
    ones(100),
    nlp,
    objective = 1.0,
    dual_feas = 1e-12,
    iter = 10,
    solver_specific = Dict(
      :matvec => 10,
      :dot => 25,
      :empty_vec => [],
      :small_vec => [2.0; 3.0],
      :axpy => 20,
      :ray => -1 ./ (1:100),
    ),
  )

  io = IOBuffer()
  show(io, output)
  @test String(take!(io)) ==
        "Solver output of type OptSolverOutput{Float64, Vector{Float64}}\nStatus: first-order stationary\n"

  @testset "Testing inference" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x -> dot(x, x), ones(T, 2))

      output = OptSolverOutput(:first_order, nlp.meta.x0, nlp)
      @test output.status == :first_order
      @test typeof(output.objective) == T
      @test typeof(output.dual_feas) == T
      @test typeof(output.primal_feas) == T

      nlp = ADNLPModel(x -> dot(x, x), ones(T, 2), x -> [sum(x) - 1], [0.0], [0.0])

      output = OptSolverOutput(:first_order, nlp.meta.x0, nlp)
      @test output.status == :first_order
      @test typeof(output.objective) == T
      @test typeof(output.dual_feas) == T
      @test typeof(output.primal_feas) == T
    end
  end

  @testset "Test throws" begin
    @test_throws Exception OptSolverOutput(:bad, nlp)
    @test_throws Exception OptSolverOutput(:unkwown, nlp, bad = true)
  end

  @testset "Testing Dummy Solver with multi-precision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x -> dot(x, x), ones(T, 2))
      solver = DummySolver(nlp)
      output = with_logger(NullLogger()) do
        solve!(solver, nlp)
      end
      @test typeof(output.objective) == T
      @test typeof(output.dual_feas) == T
      @test typeof(output.primal_feas) == T
      @test eltype(output.solution) == T
      @test eltype(output.multipliers) == T
      @test eltype(output.multipliers_L) == T
      @test eltype(output.multipliers_U) == T

      nlp = ADNLPModel(x -> dot(x, x), ones(T, 2), x -> [sum(x) - 1], [0.0], [0.0])

      solver = DummySolver(nlp)
      output = with_logger(NullLogger()) do
        solve!(solver, nlp)
      end
      @test typeof(output.objective) == T
      @test typeof(output.dual_feas) == T
      @test typeof(output.primal_feas) == T
      @test eltype(output.solution) == T
      @test eltype(output.multipliers) == T
      @test eltype(output.multipliers_L) == T
      @test eltype(output.multipliers_U) == T
    end
  end
end
