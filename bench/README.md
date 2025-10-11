# Benchmark: TRON subsolver comparison (CG vs MINRES)

This benchmark compares `tron(..., subsolver = :cg)` with `tron(..., subsolver = :minres)` using Krylov.jl.

## Requirements

- Julia 1.10 (as in Project.toml)
- From the package root, run `julia --project=.` to use the package environment. Ensure Krylov.jl is available in the environment (Project.toml pins Krylov = "0.10.0").

## Run

```sh
julia --project=. bench/bench_tron_minres_vs_cg.jl
```

## Output

- CSV printed to stdout with columns: prob,subsolver,elapsed,iter,status,objective,dual

## Notes

- The script uses synthetic logistic-like problems. Adjust sizes in the script if you want larger/smaller tests or longer time limits.
- If Krylov.jl does not support `:minres` with your installed version, you may need to update the environment to a Krylov release with MINRES trust-region support.
