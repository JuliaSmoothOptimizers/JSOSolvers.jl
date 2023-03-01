using Pkg
Pkg.activate(".")
using JSOSolvers, NLPModels, NLPModelsTest, SolverCore
Pkg.status()

nlp = NLPModelsTest.BROWNDEN()

stats, solver = GenericExecutionStats(nlp), TronSolver(nlp)

nlp.meta.x0 .= [
  -11.594439937319194
  13.203630064315156
  -0.40343918517229116
  0.2367776376028016
]
@allocated solve!(solver, nlp, stats) # , x = x0, λ = λ)
@show @allocated solve!(solver, nlp, stats) # , x = x0, λ = λ) # 0
reset!(nlp); solve!(solver, nlp, stats, verbose = 1)

nlp.meta.x0 .= [25.0; 5; -5; 1]
@allocated solve!(solver, nlp, stats)
@show @allocated solve!(solver, nlp, stats) # 0
reset!(nlp); solve!(solver, nlp, stats, verbose = 1)

#= TODO:
- Use updated SolverCore version (merge solver_specific PR + new release)
- Fix the initial Lagrange multiplier (right now it is [] and then we eval it)
=#

using Profile
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1 solve!(solver, nlp, stats)
using PProf
PProf.Allocs.pprof(from_c = false)

#= February, 28th
┌ Error: Cauchy step returned: small_step
└ @ JSOSolvers C:\Users\tangi\Documents\cvs\JSOSolvers.jl\src\tron.jl:253
┌ Error: Cauchy step returned: small_step
└ @ JSOSolvers C:\Users\tangi\Documents\cvs\JSOSolvers.jl\src\tron.jl:253
#= C:\Users\tangi\Documents\cvs\JSOSolvers.jl\test_allocs.jl:17 =# @allocated(solve!(solver, nlp, stats)) = 190952
[ Info:   iter      f(x)         π         Δ     ratio         cgstatus  
[ Info:      0   8.6e+04   1.5e-03   1.0e+00   8.2e-01  stationary point found
[ Info:      1   8.6e+04   4.0e-05   1.3e-06   1.9e-03  stationary point found
[ Info:      2   8.6e+04   1.6e-06   3.2e-07   1.7e-06  stationary point found
[ Info:      3   8.6e+04   1.6e-06   1.1e-10   8.8e-07  on trust-region boundary
[ Info:      4   8.6e+04   1.6e-06   3.2e-11   3.7e-07  on trust-region boundary
[ Info:      5   8.6e+04   1.6e-06   1.6e-11   2.5e-07  on trust-region boundary
[ Info:      6   8.6e+04   1.6e-06   7.9e-12   7.8e-08  on trust-region boundary
[ Info:      7   8.6e+04   1.6e-06   2.4e-12   3.2e-08  on trust-region boundary
[ Info:      8   8.6e+04   1.6e-06   9.9e-13   9.4e-09  on trust-region boundary
[ Info:      9   8.6e+04   1.6e-06   2.9e-13   3.7e-09  on trust-region boundary
[ Info:     10   8.6e+04   1.6e-06   1.1e-13   1.0e-09  on trust-region boundary
[ Info:     11   8.6e+04   1.6e-06   3.2e-14   3.9e-10  on trust-region boundary
[ Info:     13   8.6e+04   1.6e-06   3.3e-15   3.9e-11  on trust-region boundary
[ Info:     14   8.6e+04   1.6e-06   1.7e-15   2.6e-11  on trust-region boundary
[ Info:     15   8.6e+04   1.6e-06   8.3e-16   7.9e-12  on trust-region boundary
[ Info:     16   8.6e+04   1.6e-06   4.2e-16   4.2e-12  on trust-region boundary
[ Info:     17   8.6e+04   1.6e-06   2.1e-16   2.7e-12  on trust-region boundary
┌ Error: Cauchy step returned: small_step
└ @ JSOSolvers C:\Users\tangi\Documents\cvs\JSOSolvers.jl\src\tron.jl:253
[ Info:     18   8.6e+04   1.6e-06   1.0e-16
#= C:\Users\tangi\Documents\cvs\JSOSolvers.jl\test_allocs.jl:22 =# @allocated(solve!(solver, nlp, stats)) = 76512
[ Info:   iter      f(x)         π         Δ     ratio         cgstatus
[ Info:      0   1.7e+06   2.1e+06   1.0e+02   1.2e+00  stationary point found
[ Info:      1   4.9e+05   5.8e+05   1.3e+01   1.2e+00  stationary point found
[ Info:      2   2.0e+05   1.5e+05   1.3e+01   1.3e+00  stationary point found
[ Info:      3   9.4e+04   4.6e+04   1.3e+01   1.2e+00  stationary point found
[ Info:      4   8.6e+04   1.7e+04   1.3e+01   9.5e-01  stationary point found
[ Info:      5   8.6e+04   1.7e+03   1.3e+01   1.0e+00  stationary point found
[ Info:      6   8.6e+04   4.7e+01   1.3e+01   1.0e+00  stationary point found
[ Info:      7   8.6e+04   3.4e-01   1.3e+01   1.0e+00  stationary point found
[ Info:      8   8.6e+04   6.5e-03   1.3e+01
Analyzing 1024 allocation samples... 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:41        
"alloc-profile.pb.gz"
=#