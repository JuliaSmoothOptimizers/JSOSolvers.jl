# Benchmarks

## CUTEst benchmark

With JSO-compliant solvers, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools.
We are following here the tutorial in [SolverBenchmark.jl](https://jso.dev/SolverBenchmark.jl/stable/) to run benchmarks on JSO-compliant solvers.

``` @example ex1
using CUTEst
```

To test the implementation on bound-constrained problems, we use the package [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl), which implements `CUTEstModel` an instance of `AbstractNLPModel`.

``` @example ex1
using SolverBenchmark
```

Let us select bound-constrained problems from CUTEst with a maximum of 300 variables.

``` @example ex1
nmax = 300
pnames_unconstrained = CUTEst.select_sif_problems(
  max_con = 0,
  only_free_var = true, # unconstrained
  max_var = nmax,
  objtype = 3:6,
)
pnames = CUTEst.select_sif_problems(
  max_con = 0,
  max_var = nmax,
  objtype = 3:6,
)

pnames = setdiff(pnames, pnames_unconstrained)
cutest_problems = (CUTEstModel(p) for p in pnames)

length(cutest_problems) # number of problems
```

We compare here TRON from JSOSolvers.jl with [Ipopt](https://link.springer.com/article/10.1007/s10107-004-0559-y) (Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical programming, 106(1), 25-57.), via the [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) thin wrapper, on a subset of CUTEst problems.

``` @example ex1
using JSOSolvers, NLPModelsIpopt
```

 To make stopping conditions comparable, we set `Ipopt`'s parameters `dual_inf_tol=Inf`, `constr_viol_tol=Inf` and `compl_inf_tol=Inf` to disable additional stopping conditions related to those tolerances, `acceptable_iter=0` to disable the search for an acceptable point.

``` @example ex1
#Same time limit for all the solvers
max_time = 1200. #20 minutes
tol = 1e-5

solvers = Dict(
  :ipopt => nlp -> ipopt(
    nlp,
    print_level = 0,
    dual_inf_tol = Inf,
    constr_viol_tol = Inf,
    compl_inf_tol = Inf,
    acceptable_iter = 0,
    max_cpu_time = max_time,
    x0 = nlp.meta.x0,
    tol = tol,
  ),
  :tron => nlp -> tron(
    nlp,
    max_time = max_time,
    max_iter = typemax(Int64),
    max_eval = typemax(Int64),
    atol = tol,
    rtol = tol,
  ),
)

stats = bmark_solvers(solvers, cutest_problems)
```

The function `bmark_solvers` return a `Dict` of `DataFrames` with detailed information on the execution. This output can be saved in a data file.

``` @example ex1
using JLD2
@save "ipopt_dcildl_$(string(length(pnames))).jld2" stats
```

The result of the benchmark can be explored via tables,

``` @example ex1
pretty_stats(stats[:tron])
```

or it can also be used to make performance profiles.

``` @example ex1
using Plots
gr()

legend = Dict(
  :neval_obj => "number of f evals",
  :neval_grad => "number of ∇f evals",
  :neval_hprod  => "number of ∇²f*v evals",
  :neval_hess  => "number of ∇²f evals",
  :elapsed_time => "elapsed time"
)
perf_title(col) = "Performance profile on CUTEst w.r.t. $(string(legend[col]))"

styles = [:solid,:dash,:dot,:dashdot] #[:auto, :solid, :dash, :dot, :dashdot, :dashdotdot]

function print_pp_column(col::Symbol, stats)

  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))
  first_order(df) = df.status .== :first_order
  unbounded(df) = df.status .== :unbounded
  solved(df) = first_order(df) .| unbounded(df)
  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)

  p = performance_profile(
    stats,
    cost,
    title=perf_title(col),
    legend=:bottomright,
    linestyles=styles
  )
end

print_pp_column(:elapsed_time, stats) # with respect to time
```

``` @example ex1
print_pp_column(:neval_obj, stats) # with respect to number of objective evaluations
```
``` @example ex1
print_pp_column(:neval_grad, stats) # with respect to number of gradient evaluations
```
