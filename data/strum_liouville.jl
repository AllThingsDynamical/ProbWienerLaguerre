using OrdinaryDiffEq, Random
using Measures
using Plots
using LaTeXStrings
using Random
using NPZ

# Global publication theme
default(
    fontfamily = "Computer Modern",
    linewidth = 2.2,
    markersize = 4,
    legendfontsize = 11,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 14,
    framestyle = :box,
    grid = false,
    minorgrid = false,
    tickdirection = :out,
    foreground_color_border = :black,
    foreground_color_axis = :black,
    foreground_color_text = :black,
    background_color = :white,
    size = (720, 480),
    dpi = 300
)

# Consistent color cycle (colorblind-safe, print-friendly)
const PUB_COLORS = [
    RGB(0.0, 0.2, 0.6),   # deep blue
    RGB(0.8, 0.2, 0.2),   # red
    RGB(0.2, 0.6, 0.2),   # green
    RGB(0.6, 0.4, 0.0),   # ochre
    RGB(0.4, 0.2, 0.6)    # purple
]
palette(PUB_COLORS)

# Convenience wrapper for axis labels (LaTeX by default)
xlabel!(s) = xlabel!(L"$s$")
ylabel!(s) = ylabel!(L"$s$")



function make_input(; K=5, ω0=1.0, seed=0)
    rng = MersenneTwister(seed)
    ω = ω0 .* collect(1:K)
    ϕ = 2π .* rand(rng, K)
    c = ones(K) ./ sqrt(K)
    t -> sum(c[k] * sin(ω[k]*t + ϕ[k]) for k in 1:K)
end

function simulate_second_order(; a=0.8, b=4.0, c=1.2,
                               T=50.0, dt=0.01,
                               y0=0.0, v0=0.0,
                               seed=0, noise_std=0.1)
    u = make_input(; seed=seed)
    rng = MersenneTwister(seed)

    function f!(du, x, p, t)
        du[1] = x[2]
        du[2] = -a*x[2] - b*x[1] + c*u(t)
    end

    prob = ODEProblem(f!, [y0, v0], (0.0, T))
    sol  = solve(prob, Tsit5(); saveat=dt)

    t = sol.t
    y = sol[1, :] .+ noise_std .* randn(rng, length(sol.t))
    return t, u.(t), y
end

t, u, y = simulate_second_order()
figure1 = plot(t,u, xlabel="t", label="u")
figure2 = plot(t,y, xlabel="t", label="y")
plot(figure1, figure2, size=(1000, 300), margin=5mm)
savefig("data/strum_l.png")

u_train = u[1:2500]
u_test = u[2501:end]
y_train = y[1:2500]
y_test = y[2501:end]

file = npzwrite("data/strum_l.npz",
Dict(
    "U_train" => u_train,
    "U_test" => u_test,
    "Y_train" => y_train,
    "Y_test" => y_test
))