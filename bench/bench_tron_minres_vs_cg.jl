using Pkg, LinearAlgebra, Printf, Dates
using JSOSolvers, Random, ADNLPModels

function make_logistic(n, m; rng = Random.GLOBAL_RNG)
  A = randn(rng, m, n)
  xtrue = randn(rng, n)
  y = sign.(A * xtrue + 0.1 * randn(rng, m))
  f(x) = begin
    z = A * x
    s = @. log(1 + exp(-y * z))
    sum(s)
  end
  gradf(x) = begin
    z = A * x
    sig = @. 1 / (1 + exp(y * z))
    return A' * (-y .* sig)
  end
  x0 = zeros(n)
  l = fill(-Inf, n)
  u = fill(Inf, n)
  model = ADNLPModel((x->f(x)), x0, l, u; grad = (x->gradf(x)))
  return model
end

function run_one(model; subsolver = :cg, max_time = 10.0)
  t0 = time()
  stats = tron(model, subsolver = subsolver, max_time = max_time, verbose = 0, subsolver_verbose = 0)
  t = time() - t0
  return (subsolver = subsolver, elapsed = t, iter = stats.iter, status = stats.status, objective = stats.objective, dual = stats.dual_feas)
end

function main()
  rng = MersenneTwister(1234)
  problems = [make_logistic(1000, 2000, rng = rng), make_logistic(2000, 4000, rng = rng)]

  results = []
  for (i, model) in enumerate(problems)
    for sub in (:cg, :minres)
      push!(results, merge((prob = i,), run_one(model, subsolver = sub, max_time = 60.0)))
    end
  end

  println("prob,subsolver,elapsed,iter,status,objective,dual")
  for r in results
    @printf("%d,%s,%.6f,%d,%s,%.12g,%.6g\n", r[:prob], string(r[:subsolver]), r[:elapsed], r[:iter], string(r[:status]), r[:objective], r[:dual])
  end
end

main()
