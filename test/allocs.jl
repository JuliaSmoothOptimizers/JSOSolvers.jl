"""
    @wrappedallocs(expr)
Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).
For example, `@wrappedallocs(x + y)` produces:
```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```
You can use this macro in a unit test to verify that a function does not
allocate:
```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

if Sys.isunix()
  @testset "Allocation tests" begin
    @testset "$symsolver" for symsolver in
                              (:LBFGSSolver, :FoSolver, :FomoSolver, :TrunkSolver, :TronSolver)
      for model in NLPModelsTest.nlp_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (symsolver == :TronSolver))
          if (symsolver == :FoSolver || symsolver == :FomoSolver)
            solver = eval(symsolver)(nlp; M = 2) # nonmonotone configuration allocates extra memory
          else
            solver = eval(symsolver)(nlp)
          end
          if symsolver == :FomoSolver
            T = eltype(nlp.meta.x0)
            stats = GenericExecutionStats(nlp, solver_specific = Dict(:avgβmax => T(0)))
          else
            stats = GenericExecutionStats(nlp)
          end
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            SolverCore.reset!(solver)
            NLPModels.reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end

    @testset "$name" for (name, symsolver) in (
      (:TrunkSolverNLS, :TrunkSolverNLS),
      (:R2SolverNLS, :R2SolverNLS),
      (:R2SolverNLS_QRMumps, :R2SolverNLS),
      (:TronSolverNLS, :TronSolverNLS),
    )
      for model in NLPModelsTest.nls_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (symsolver == :TronSolverNLS))
          if name == :R2SolverNLS_QRMumps
            solver = eval(symsolver)(nlp, subsolver = :qrmumps)
          else
            solver = eval(symsolver)(nlp)
          end

          stats = GenericExecutionStats(nlp)
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            SolverCore.reset!(solver)
            NLPModels.reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end
  end
end
