if CUDA.functional()
  @testset "GPU multiple precision support tests (NLP)" begin
    @testset "GPU multiple precision support of problem $p with $fun (NLP)" for p in
                                                                                NLPModelsTest.nlp_problems,
      fun in [trunk; fomo]

      nlp = eval(Symbol(p))(CuArray{Float64, 1, CUDA.DeviceMemory})
      if !(unconstrained(nlp))
        continue
      end
      CUDA.allowscalar() do
        stats = fun(nlp)
        @test stats.status == :first_order
      end
    end

    @testset "GPU multiple precision support of problem $p with $fun (NLS)" for p in
                                                                                NLPModelsTest.nls_problems,
      fun in [trunk]

      nls = eval(Symbol(p))(CuArray{Float64, 1, CUDA.DeviceMemory})
      if !(unconstrained(nls))
        continue
      end
      CUDA.allowscalar() do
        stats = fun(nls)
        @test stats.status == :first_order
      end
    end
  end
end
