@testset "Callback" begin
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])

  cb = (nlp, solver, wks) -> begin
    if wks[:iter] == 7
      wks[:user_stop] = true
    end
  end

  @testset "Solver $solver" for solver in [trunk]
    reset!(nlp)
    output = solver(nlp)
    @test output.iter > 7

    reset!(nlp)
    output = solver(nlp, callback=cb)
    @test output.iter == 7
  end
end