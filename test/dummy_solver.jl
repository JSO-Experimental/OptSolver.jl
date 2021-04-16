mutable struct DummySolver{T, S} <: AbstractOptSolver{T, S}
  initialized::Bool
  params::Dict
  workspace
end

SolverCore.problem_types_handled(::Type{DummySolver}) = [:unc, :bnd, :equ, :bndequ, :ineq, :genopt]

function SolverCore.parameters(::Type{DummySolver{T, S}}) where {T, S}
  (
    α = (default = T(1e-2), type = :log, min = √√eps(T), max = one(T) / 2),
    β = (default = T(1e-2), type = :log, min = √√eps(T), max = one(T) / 2),
    δ = (default = √eps(T), type = :log, min = √eps(T), max = √√√eps(T)),
    reboot_y = (default = false, type = :bool),
  )
end

function SolverCore.are_valid_parameters(::Type{DummySolver}, α, β, δ, reboot_y)
  return α + β ≤ 0.5
end

function DummySolver(
  meta::AbstractNLPModelMeta;
  T = eltype(meta.x0),
  S = typeof(meta.x0),
  kwargs...,
)
  nvar, ncon = meta.nvar, meta.ncon
  params = parameters(DummySolver{T, S})
  solver = DummySolver{T, S}(
    true,
    Dict(k => v[:default] for (k, v) in pairs(params)),
    ( # workspace
      x = S(undef, nvar),
      xt = S(undef, nvar),
      gx = S(undef, nvar),
      dual = S(undef, nvar),
      y = S(undef, ncon),
      cx = S(undef, ncon),
      ct = S(undef, ncon),
    ),
  )
  for (k, v) in kwargs
    solver.params[k] = v
  end
  solver
end

function SolverCore.solve!(
  solver::DummySolver{T, S},
  nlp::AbstractNLPModel;
  x::S = nlp.meta.x0,
  atol::Real = sqrt(eps(T)),
  rtol::Real = sqrt(eps(T)),
  max_eval::Int = 1000,
  max_time::Float64 = 30.0,
  kwargs...,
) where {T, S}
  # Check dim
  solver.initialized || error("Solver not initialized.")
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  for (k, v) in kwargs
    solver.params[k] = v
  end
  α = solver.params[:α]
  β = solver.params[:β]
  δ = solver.params[:δ]
  reboot_y = solver.params[:reboot_y]

  start_time = time()
  elapsed_time = 0.0
  solver.workspace.x .= x # Copy values
  x = solver.workspace.x  # Change reference

  cx = solver.workspace.cx .= ncon > 0 ? cons(nlp, x) : zeros(T, 0)
  ct = solver.workspace.ct .= zero(T)
  grad!(nlp, x, solver.workspace.gx)
  gx = solver.workspace.gx
  Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
  y = solver.workspace.y .= -Jx' \ gx
  Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)

  dual = solver.workspace.dual .= gx .+ Jx' * y

  iter = 0

  ϵd = atol + rtol * norm(dual)
  ϵp = atol

  ϕ(fx, cx, y) = fx + norm(cx)^2 / 2δ + dot(y, cx)
  fx = obj(nlp, x)
  @info log_header([:iter, :f, :c, :dual, :t], [Int, T, T, Float64])
  @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  solved = norm(dual) < ϵd && norm(cx) < ϵp
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

  while !(solved || tired)
    Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)
    W = Symmetric([Hxy zeros(T, nvar, ncon); Jx -δ*I], :L)
    Δxy = -W \ [dual; cx]
    Δx = Δxy[1:nvar]
    Δy = Δxy[(nvar + 1):end]

    AΔx = Jx * Δx
    ϕx = ϕ(fx, cx, y)
    xt = solver.workspace.xt .= x + Δx
    if ncon > 0
      cons!(nlp, xt, ct)
    end
    ft = obj(nlp, xt)
    slope = -dot(Δx, Hxy, Δx) - norm(AΔx)^2 / δ
    t = 1.0
    while !(ϕ(ft, ct, y) ≤ ϕx + (α + β) * t * slope)
      t /= 2
      xt .= x + t * Δx
      if ncon > 0
        cons!(nlp, xt, ct)
      end
      ft = obj(nlp, xt)
    end

    x .= xt

    fx = ft
    grad!(nlp, x, gx)
    Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
    if reboot_y
      y .= -Jx' \ gx
    else
      y .+= t * Δy
    end
    cx .= ct
    dual .= gx .+ Jx' * y
    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  end

  status = if solved
    :first_order
  elseif elapsed_time > max_time
    :max_time
  else
    :max_eval
  end

  return OptSolverOutput(
    status,
    x,
    nlp,
    objective = fx,
    dual_feas = norm(dual),
    primal_feas = norm(cx),
    multipliers = y,
    multipliers_L = zeros(T, nvar),
    multipliers_U = zeros(T, nvar),
    elapsed_time = elapsed_time,
    iter = iter,
  )
end
