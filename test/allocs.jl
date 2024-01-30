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
    @testset "$symsolver" for symsolver in (:LBFGSSolver, :R2Solver, :TrunkSolver, :TronSolver)
      for model in NLPModelsTest.nlp_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (symsolver == :TronSolver))
          solver = eval(symsolver)(nlp)
          stats = GenericExecutionStats(nlp)
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            reset!(solver)
            reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end

    @testset "$symsolver" for symsolver in (:TrunkSolverNLS, :TronSolverNLS)
      for model in NLPModelsTest.nls_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (symsolver == :TronSolverNLS))
          solver = eval(symsolver)(nlp)
          stats = GenericExecutionStats(nlp)
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            reset!(solver)
            reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end
  end
end
