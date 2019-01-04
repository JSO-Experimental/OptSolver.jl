using Optimize
using OptimizationProblems
using JuMP
using NLPModels
using NLPModelsJuMP
using Printf

probs = filter(name -> name != :OptimizationProblems, names(OptimizationProblems))
n = 5

for prob in probs
  @info @sprintf("Checking %s: ", string(prob))
  nlp = MathProgNLPModel(eval(prob)(n));
  @info @sprintf("%d variables and %d constraints\n", nlp.meta.nvar, nlp.meta.ncon)

  g_errs = gradient_check(nlp)
  for k in keys(g_errs)
    @info @sprintf(" objective: error in %d-th derivative = %7.1e\n", k, g_errs[k])
  end

  J_errs = jacobian_check(nlp)
  for ij in keys(J_errs)
    i, j = ij
    @info @sprintf(" constraint %d: error in %d-th derivative = %7.1e\n", j, i, J_errs[ij])
  end

  H_errs = hessian_check(nlp)
  for k in keys(H_errs)
    for ij in keys(H_errs[k])
      i, j = ij
      @info @sprintf(" function %d: error in (%d,%d) second derivative = %7.1e\n", k, i, j, H_errs[k][ij])
    end
  end

  H_errs = hessian_check_from_grad(nlp)
  for k in keys(H_errs)
    for ij in keys(H_errs[k])
      i, j = ij
      @info @sprintf(" function %d: error in (%d,%d) second derivative = %7.1e\n", k, i, j, H_errs[k][ij])
    end
  end
end
