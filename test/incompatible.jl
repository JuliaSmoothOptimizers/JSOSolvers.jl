mutable struct DummyModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
end

mutable struct DummyNLSModel{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
end

function test_incompatible()
  @testset "Incompatible problems should throw error" begin
    solvers = Dict(:lbfgs => [:unc], :trunk => [:unc], :tron => [:unc, :bnd])
    problems = Dict(
      :unc => ADNLPModel(x -> 0, zeros(2)),
      :bnd => ADNLPModel(x -> 0, zeros(2), zeros(2), ones(2)),
      :equ => ADNLPModel(x -> 0, zeros(2), x -> [0.0], [0.0], [0.0]),
      :ine => ADNLPModel(x -> 0, zeros(2), x -> [0.0], [-Inf], [0.0]),
      :gen => ADNLPModel(x -> 0, zeros(2), x -> [0.0; 0.0], [-Inf; 0.0], zeros(2)),
    )
    for (ptype, problem) in problems, (solver, types_accepted) in solvers
      @testset "Testing that $solver on problem type $ptype raises error: " begin
        if !(ptype in types_accepted)
          @test_throws ErrorException eval(solver)(problem)
        end
      end
    end
    nlp = DummyModel(NLPModelMeta(1, minimize = false))
    @testset for solver in keys(solvers)
      @test_throws ErrorException eval(solver)(nlp)
    end

    solvers = Dict(:trunk => [:unc], :tron => [:unc, :bnd])
    problems = Dict(
      :unc => ADNLSModel(x -> [0], zeros(2), 1),
      :bnd => ADNLSModel(x -> [0], zeros(2), 1, zeros(2), ones(2)),
    )
    for (ptype, problem) in problems, (solver, types_accepted) in solvers
      @testset "Testing that $solver on problem type $ptype raises error: " begin
        if !(ptype in types_accepted)
          @test_throws ErrorException eval(solver)(problem)
        end
      end
    end
    nls = DummyNLSModel(NLPModelMeta(1, minimize = false))
    @testset for solver in keys(solvers)
      @test_throws ErrorException eval(solver)(nls)
    end
  end
end

test_incompatible()
