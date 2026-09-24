using QRMumps

@testset "qr_mumps input-size guards" begin
  nls = ADNLSModel(x -> [x[1] - 1; 2 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  @test_throws ArgumentError QRMumpsSubsolver(nls; max_nvar = -1)
  @test_throws ArgumentError QRMumpsSubsolver(nls; max_nnzj = -1)
  @test_throws ArgumentError QRMumpsSubsolver(nls; dense_max_entries = -1)
  for limits in (
    (max_nvar = 1,),
    (max_nnzj = nls.nls_meta.nnzj - 1,),
    # dense 2x2 Jacobian rejected only once the density guard is armed
    (dense_max_entries = 0,),
  )
    sub = QRMumpsSubsolver(nls; limits...)
    @test JSOSolvers.is_unsupported(sub)
    @test sub.spmat === nothing && sub.spfct === nothing
    @test isempty(sub.irn) && isempty(sub.jcn) && isempty(sub.val)
    stats = @test_logs (:error, r"skipping qr_mumps") R2NLS(nls; subsolver = sub)
    @test stats.status == :exception
    @test stats.iter == 0
  end
end

@testset "Testing qr_mumps Subsolver density threshold" begin
  nls = ADNLSModel(x -> [x[1] - 1; 2 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  @test !is_unsupported(QRMumpsSubsolver(nls))

  # The threshold counts dense Jacobian entries, so problems whose dense `m * n`
  # fits under it are handed to qr_mumps no matter how dense they are.
  emax = JSOSolvers.DEFAULT_QRMUMPS_DENSE_MAX_ENTRIES
  @test emax == JSOSolvers.DEFAULT_HSL_DENSE_MAX_NVAR^2
  @test JSOSolvers.dense_jacobian_entries(5_000, 5_000) == 25_000_000
  for (m, n) in ((2, 2), (500, 500), (5_000, 5_000), (12_000, 12_000), (1_000_000, 100))
    @test !JSOSolvers.qrmumps_guard_triggered(
      m,
      n,
      JSOSolvers.dense_jacobian_entries(m, n),
      0.5,
      emax,
      typemax(Int),
      typemax(Int),
    )
  end

  # A tall, skinny Jacobian is caught on entries even though `nvar` is small --
  # this is why the qr_mumps threshold is on `m * n`, not on `nvar` alone.
  for (m, n) in ((12_001, 12_001), (1_000_000, 200))
    @test JSOSolvers.dense_jacobian_entries(m, n) > emax
    @test JSOSolvers.qrmumps_guard_triggered(
      m,
      n,
      JSOSolvers.dense_jacobian_entries(m, n),
      0.5,
      emax,
      typemax(Int),
      typemax(Int),
    )
    # ... but a genuinely sparse Jacobian of the same shape still goes to qr_mumps
    @test !JSOSolvers.qrmumps_guard_triggered(m, n, 5n, 0.5, emax, typemax(Int), typemax(Int))
  end

  # A dense problem below the threshold is actually solved, not skipped. Every
  # residual component depends on every variable, so the Jacobian is structurally
  # dense; the trailing row makes the system inconsistent, leaving a nonzero
  # residual at the solution.
  dense_n = 30
  dense_nls =
    ADNLSModel(x -> vcat(sum(x) .+ x .- 1, 3 * sum(x) + 2), ones(dense_n), dense_n + 1)
  dense_m = dense_nls.nls_meta.nequ
  @test dense_nls.nls_meta.nnzj > 0.5 * JSOSolvers.dense_jacobian_entries(dense_m, dense_n)
  dense_sub = QRMumpsSubsolver(dense_nls)
  @test !is_unsupported(dense_sub)
  @test R2NLS(dense_nls; subsolver = dense_sub).status == :first_order
end
