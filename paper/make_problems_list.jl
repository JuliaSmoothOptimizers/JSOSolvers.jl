using Pkg
path = dirname(@__FILE__)
Pkg.activate(path)
using CUTEst

nmax = 10000
pnames = CUTEst.select(
  max_con = 0,
  only_free_var = true, # unconstrained
  max_var = nmax,
  objtype = 3:6,
)

open("list_problems_$nmax.dat", "w") do io
  for name in pnames
    write(io, name * "\n")
  end
end
