const opt_problem_type = [:unc, :bnd, :equ, :bndequ, :ineq, :genopt]

# I was hoping to add these to the metaprogramming code below, but failed
can_solve_unc(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :unc)
can_solve_bnd(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :bnd)
can_solve_equ(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :equ)
can_solve_bndequ(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :bndequ)
can_solve_ineq(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :ineq)
can_solve_optgen(::Type{S}) where {S <: AbstractSolver} = can_solve_type(S, :optgen)

for ptype in opt_problem_type
  fname = Symbol("can_solve_$ptype")
  help = """
      $fname(solver)

  Check if the `solve` can solve optimization problems of type `$ptype`.
  Call `problem_types_handled` for a list of problem types that the `solver` can solve.
  """
  @eval begin
    @doc $help $fname
    export $fname
  end
end
