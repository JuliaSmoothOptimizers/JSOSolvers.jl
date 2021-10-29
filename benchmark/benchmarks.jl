using BenchmarkTools
# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools, ADNLPModels, SolverTest, JSOSolvers

const SUITE = BenchmarkGroup()
SUITE[:lbfgs] = BenchmarkGroup()
n = 30
D = Diagonal([0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n])
A = spdiagm(0 => 2 * ones(n), -1 => -ones(n - 1), -1 => -ones(n - 1))


function benchmark_lbfgs()
  # unconstrained problems found in Solvertest:
  a₁ = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 1)^2, zeros(2), name = "(x₁ - 1)² + 4(x₂ - 1)²")
  a₂ = ADNLPModel(x -> dot(x .- 1, D, x .- 1), zeros(n), name = "Diagonal quadratic")
  a₃ = ADNLPModel(x -> dot(x .- 1, A, x .- 1), zeros(n), name = "Tridiagonal quadratic")
  a₄ = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0], name = "Rosenbrock")
  a₅ = ADNLPModel(
      x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
      collect(1:n) ./ (n + 1),
      name = "Extended Rosenbrock",
    )

  ad_problems = [a₁, a₂, a₃, a₄, a₅]
  for i in 1:5
    SUITE[:lbfgs][i] = @benchmarkable [lbfgs(nlp) for nlp in $ad_problems]
  end
end

benchmark_lbfgs()
