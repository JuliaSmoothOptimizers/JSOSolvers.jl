function test_incompatible()
  solvers = Dict(:lbfgs => [:unc],
                 :trunk => [:unc],
                 :tron  => [:unc, :bnd])
  problems = Dict(:unc => ADNLPModel(x -> 0, zeros(2)),
                  :bnd => ADNLPModel(x -> 0, zeros(2), lvar=zeros(2), uvar=ones(2)),
                  :equ => ADNLPModel(x -> 0, zeros(2), c=x->[0.0], lcon=[0.0], ucon=[0.0]),
                  :ine => ADNLPModel(x -> 0, zeros(2), c=x->[0.0], lcon=[-Inf], ucon=[0.0]),
                  :gen => ADNLPModel(x -> 0, zeros(2), c=x->[0.0; 0.0], lcon=[-Inf; 0.0], ucon=zeros(2))
                 )
  for (ptype, problem) in problems, (solver, types_accepted) in solvers
    @testset "Testing that $solver on problem type $ptype raises error: " begin
      if !(ptype in types_accepted)
        @test_throws ErrorException eval(solver)(problem)
      end
    end
  end
end

test_incompatible()
