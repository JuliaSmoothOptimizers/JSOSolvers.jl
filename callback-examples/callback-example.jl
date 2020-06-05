using JSOSolvers, LaTeXStrings, Logging, NLPModels, Plots, ADNLPModels
gr(size=(800,800))

function callback_example()
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  xg = range(-1.5, 1.5, length=50)
  yg = range(-1.5, 1.5, length=50)

  X = [nlp.meta.x0[1]]
  Y = [nlp.meta.x0[2]]

  function cb(nlp, solver, wks)
    x = solver.x
    push!(X, x[1])
    push!(Y, x[2])

    title = "Iter $(wks[:iter]) - ‖∇f‖ = $(norm(solver.gx))"
    plt = plot(leg=false, title=title, xlabel=L"x_1", ylabel=L"x_2")
    contour!(xg, yg, (x1,x2) -> f([x1; x2]), levels=100)
    plot!(X, Y, c=:red, l=:arrow)
    scatter!(X, Y, c=:red)
    xlims!(extrema(xg)...)
    ylims!(extrema(yg)...)
    display(plt)

    sleep(0.5)
  end

  output = with_logger(NullLogger()) do
    trunk(nlp, callback=cb)
  end
end

callback_example()