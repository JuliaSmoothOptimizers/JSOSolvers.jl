# Multiple Floating-Point Systems Support


The following example also illustrates that the solvers are compatible with most data structures supported by Julia by running the solver TRUNK on GPUs.
We use here ExaModels.jl [@shin2024accelerating] to model an optimization problem as it implements the NLPModel API and is compatible with GPU backends.

```julia
using CUDA, ExaModels, JSOSolvers, NLPModels, OptimizationProblems
problem = "woods"
nscal = 100
model = OptimizationProblems.PureJuMP.eval(Meta.parse(problem))(n = nscal)
nlp_gpu = ExaModels.ExaModel(model; backend=CUDABackend(), prod=true)
stats = trunk(nlp_gpu)
```

