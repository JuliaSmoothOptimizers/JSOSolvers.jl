# Solvers

Here is the list of solvers in this package:

- [`lbfgs`](@ref)
- [`tron`](@ref)
- [`trunk`](@ref)
- [`R2`](@ref)

Here it is in table format:

| Problem type          | Solvers  |
| --------------------- | -------- |
| Unconstrained NLP     | [`lbfgs`](@ref), [`tron`](@ref), [`trunk`](@ref), [`R2`](@ref)|
| Unconstrained NLS     | [`trunk`](@ref), [`tron`](@ref) |
| Bound-constrained NLP | [`tron`](@ref) |
| Bound-constrained NLS | [`tron`](@ref) |

## Solver list

```@docs
lbfgs
tron
trunk
R2
```
