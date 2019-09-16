using DerivativeFreeSolvers

using Logging

export tune!

include("tuning-model.jl")

"""
    tune!(solver, problems)

Optimize parameters of `solver` for `problems`. Currently restricted to
- The parameters, bounds and constraints described by the solver itself;
- The blackbox evaluation is the sum of counters of the problem weighted
by 10 if the solver declared failure for that problem.

Disclaimer: For use with CUTEst, do
```
ps = CUTEst.select(...)
problems = (CUTEstModel(p) for p in ps)
```
"""
function tune!(solver :: JSOSolver, problems :: Any;
               parameters :: Array{Symbol} = [:all],
               tol :: Real = 1e-2,
               max_eval :: Real = 100,
               solver_args :: Dict = Dict()
              )

  I = findall(solver.types .== :real)
  if parameters != [:all]
    I = I ∩ indexin(parameters, solver.params)
  end
  params = solver.params[I]
  x0, lvar, uvar = Float64.(solver.values[I]), solver.lvar[I], solver.uvar[I]
  c,  lcon, ucon = solver.cons,  solver.lcon, solver.ucon

  NP = length(params)

  @info "Parameter optimization of $solver"
  @info "  Parameters under consideration:"
  for i = 1:NP
    @info @sprintf("  %+14.8e  ≦  %10s  ≦  %+14.8e\n", lvar[i], params[i], uvar[i])
  end
  if length(lcon) > 0
    @info "  There are also constraints, which can't be described here"
  else
    @info "  No further constraints"
  end

  f(x) = begin
    s = 0.0
    try
      solver.values[I] .= x
      stats = with_logger(NullLogger()) do
        solve_problems(solver, problems; solver_args...)
      end
      s += dot(1.0 .+ (stats.status .!= :first_order) * 9, stats.neval_obj)
    catch ex
      @warn "Unhandled exception on tuning: $ex"
      s = Inf
    end

    return s
  end

  tnlp = TuningProblem(f, x0, lvar, uvar, c, lcon, ucon)
  output = mads(tnlp, max_eval=max_eval, max_time=Inf, rtol=0.0, atol=tol, extreme=true, base_decrease=1.1)

  @info("Parameter optimization finished with status $(output.status)")
  x = output.solution

  @info "Optimal parameter choice:"
  for i = 1:NP
    @info "  $(params[i]) = $(x[i])"
  end

  solver.values[I] .= x

  return solver
end
