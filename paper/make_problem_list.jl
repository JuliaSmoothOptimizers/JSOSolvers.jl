using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using CUTEst

pnames = CUTEst.select_sif_problems(
  max_con = 0,
  only_free_var = true, # unconstrained
  objtype = 3:6,
)

open("list_problems.dat", "w") do io
  for name in pnames
    write(io, name * "\n")
  end
end
