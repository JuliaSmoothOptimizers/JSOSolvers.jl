@testset "objgrad on tron" begin
  struct MyProblem <: AbstractNLPModel
    meta :: NLPModelMeta
    counters :: Counters
  end

  function MyProblem()
    meta = NLPModelMeta(
      2, # nvar
      x0 = [0.1; 0.1],
      lvar=zeros(2),
      uvar=ones(2)
    )
    MyProblem(meta, Counters())
  end

  function NLPModels.objgrad!(:: MyProblem, x :: AbstractVector, g:: AbstractVector)
    f = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
    g[1] = 2 * (x[1] - 1) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    f, g
  end

  function NLPModels.hprod!(:: MyProblem, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0)
    Hv[1] = obj_weight * (2 - 400 * (x[2] - x[1]^2) + 800 * x[1]^2) * v[1] - 400obj_weight * x[1] * v[2]
    Hv[2] = 200obj_weight * v[2] - 400obj_weight * x[1] * v[1]
    Hv
  end

  nlp = MyProblem()
  output = tron(nlp, use_only_objgrad=true)
  @test isapprox(output.solution, ones(2), rtol=1e-4)
  @test output.dual_feas < 1e-4
  @test output.objective< 1e-4

  @test_throws MethodError tron(nlp, use_only_objgrad=false)
end