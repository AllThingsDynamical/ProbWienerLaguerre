using OrdinaryDiffEq
using Plots
using LaTeXStrings
using HDF5
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


# Van der Pol oscillator in oscillatory regime (μ > 0)
function vdp!(du, u, μ, t)
    x, v = u
    du[1] = v
    du[2] = μ * (1 - x^2) * v - x
end

function simulate_vdp(; μ=2.0, u0=[2.0, 0.0], tspan=(0.0, 40.0), dt=0.01)
    prob = ODEProblem(vdp!, u0, tspan, μ)
    sol = solve(prob, Tsit5(); saveat=dt)
    return sol.t, Array(sol)
end

t, u = simulate_vdp()
eps = 1e-1
u_noisy = u + eps*randn(size(u))

figure = plot(t, u_noisy[1,:], label="u", xlabel="t")
plot!(t, u_noisy[2,:], label="v")
savefig("data/vander_pol.png")

u_train = u_noisy[:, 1:2000]
u_test = u_noisy[:, 2001:end]

file = npzwrite("data/vander-pol.npz",
Dict(
    "U_train" => u_train,
    "U_test" => u_test
))