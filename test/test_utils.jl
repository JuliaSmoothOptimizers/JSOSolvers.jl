@testset "opnorm and power-method tests (LinearOperators)" begin
  # 1) Square Float64 via direct LAPACK or ARPACK
  A_mat = [2.0 0.0; 0.0 -1.0]
  A_op = LinearOperator(A_mat)
  λ, ok = opnorm(A_op)
  @test ok == true
  @test isapprox(λ, 2.0; atol = 1e-12)

  # 2) Rectangular Float64 via direct LAPACK or ARPACK SVD
  J_mat = [3.0 0.0 0.0; 0.0 1.0 0.0]
  J_op = LinearOperator(J_mat)
  σ, ok_sv = opnorm(J_op)
  @test ok_sv == true
  @test isapprox(σ, 3.0; atol = 1e-12)

  # 3) Square BigFloat via power-method
  B_mat = Matrix{BigFloat}([2.0 0.0; 0.0 -1.0])
  B_op = LinearOperator(B_mat)
  λ_bf, ok_bf = opnorm(B_op)
  @test ok_bf == true
  @test isapprox(λ_bf, BigFloat(2); atol = 10*eps(BigFloat))

  # 4) Rectangular BigFloat via rectangular power-method
  JR_mat = Matrix{BigFloat}([3.0 0.0 0.0; 0.0 1.0 0.0])
  JR_op = LinearOperator(JR_mat)
  σ_bf, ok_bf2 = opnorm(JR_op)
  @test ok_bf2 == true
  @test isapprox(σ_bf, BigFloat(3); atol = 10*eps(BigFloat))
end
