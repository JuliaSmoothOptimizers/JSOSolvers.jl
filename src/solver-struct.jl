mutable struct JSOSolver
  solver        # Which function to call
  params :: Vector{Symbol}
  values :: Vector{<: Any}
  types  :: Vector{Symbol} # :continuous, :discrete, etc. (Not implemented)
  lvar   :: Vector{<: Real}
  uvar   :: Vector{<: Real}
  cons
  lcon   :: Vector{<: Real}
  ucon   :: Vector{<: Real}
end

function (s :: JSOSolver)(nlp :: AbstractNLPModel; kwargs...)
  return s.solver(nlp; zip(s.params, s.values)..., kwargs...)
end

function Base.getindex(solver :: JSOSolver, key :: Symbol)
  i = findfirst(solver.params .== key)
  return solver.values[i]
end

# TODO?: Enforce type
function Base.setindex!(solver :: JSOSolver, value :: Any, key :: Symbol)
  i = findfirst(solver.params .== key)
  solver.values[i] = value
end
