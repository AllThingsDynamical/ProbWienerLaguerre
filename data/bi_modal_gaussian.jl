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

using LinearAlgebra
using QuasiMonteCarlo

function generate_gaussian_data(d::Int, nsamples::Int=10_000)
    center_one = 30*ones(d)
    center_two = -30*ones(d)
    center_three = zeros(d)
    sd = 20

    function gaussian(x, y)
        exp(-0.5*(1/sd^2)*norm(x-y)^2)
    end

    lower_limits = -70*ones(d)
    upper_limits = 70*ones(d)
    samples = QuasiMonteCarlo.sample(nsamples, lower_limits, upper_limits, QuasiMonteCarlo.LatinHypercubeSample())
    X = samples

    Y = zeros(1, nsamples)
    for i=1:nsamples
        Y[1,i] = gaussian(X[:,i], center_one) + gaussian(X[:,i], center_two)+ gaussian(X[:,i], center_three)
    end

    return X, Y
end

function train_test_split(
    X::AbstractMatrix,
    Y::AbstractMatrix;
    frac_train::Float64 = 0.8,
    shuffle::Bool = true,
    rng::AbstractRNG = Random.default_rng()
)
    d, M = size(X)
    n, M2 = size(Y)
    @assert M == M2 "X and Y must have the same number of columns"

    idx = collect(1:M)
    shuffle && Random.shuffle!(rng, idx)

    M_train = floor(Int, frac_train * M)
    train_idx = idx[1:M_train]
    
    X_test, Y_test = generate_gaussian_data(d, 20_000)

    return (X[:, train_idx], Y[:, train_idx]),
           (X_test,  Y_test)
end

function write_data(
    train::Tuple,
    test::Tuple,
    filename::String
)
    Xtr, Ytr = train
    Xte, Yte = test

    npzwrite(filename, Dict(
        "X_train" => Float64.(Xtr),
        "Y_train" => Float64.(Ytr),
        "X_test"  => Float64.(Xte),
        "Y_test"  => Float64.(Yte),
    ))
    nothing
end

begin # Visualize data
    d = 2
    X, Y = generate_gaussian_data(d)
    figure1 = scatter(X[1,:], X[2,:], ms=4.0, marker_z=Y[1,:], colorbar=true, label=false
    ,xlabel="x", ylabel="y")
    savefig("data/gaussian_2d.png")
end

begin # Generate data
    dimensions = [2,3,5,10,20]
    for d in dimensions
        X, Y = generate_gaussian_data(d)
        train, test = train_test_split(X, Y)
        write_data(train, test, "data/gaussian_$d.npz")
    end
end