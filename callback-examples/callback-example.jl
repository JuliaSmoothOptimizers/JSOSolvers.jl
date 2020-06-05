using JSOSolvers, LaTeXStrings, Logging, NLPModels, Plots
pyplot(size=(800,800))

function callback_example()
  f(x) = (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  xg = range(-1.5, 1.5, length=50)
  yg = range(-1.5, 1.5, length=50)

  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]

  anim = Animation()
  function cb(nlp, wks)
    x = wks[:x]
    push!(X, x[1])
    push!(Y, x[2])

    plot(leg=false, title="Iter $(wks[:iter])", xlabel=L"x_1", ylabel=L"x_2")
    contour!(xg, yg, (x1,x2) -> f([x1; x2]), levels=100)
    plot!(X, Y, c=:red, l=:arrow)
    scatter!(X, Y, c=:red)
    xlims!(extrema(xg)...)
    ylims!(extrema(yg)...)

    frame(anim)
  end
  
  output = with_logger(NullLogger()) do
    trunk(nlp, callback=cb)
  end

  gif(anim, "trunk-on-rosenbrock.gif", fps=4)
end

callback_example()