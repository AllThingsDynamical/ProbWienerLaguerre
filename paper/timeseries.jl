using NPZ
using Plots
include("../jl-src/custom_plot.jl")
include("../jl-src/wl-model.jl")
include("../jl-src/estimators.jl")

begin
    filename = "data/vander-pol.npz"
    data = npzread(filename)
    U_train = data["U_train"]
    U_test = data["U_test"]
end

begin
    k=1
    p = 50
    λ = 3.0

    M1 = size(U_train, 2)
    resolution1 = 2*M1
    M2 = size(U_test, 2)
    resolution2 = 2*M2

    model1 = WL_Layer(k, resolution1, p, λ)
    wl_features, output_features = generate_input_output_features(U_train, model1)

    X = wl_features
    Y = output_features
    estimator = elm_estimator(X, Y, 2000, λ=5e-1)
end

begin
    W = hcat(U_train[:,500:end], U_test)
    m = 2*size(W,2)
    model_w = WL_Layer(k, m, p, λ)
    wl_features, _ = generate_input_output_features(W, model_w)
    y_pred = estimator(wl_features)
    y_pred = 4/3*(y_pred .- 0.5)

    n = size(W,2) - 1000
    y_recon = y_pred[:,1:n]
    y_future = y_pred[:, n+1:end]
    figure1 = plot(1:n, y_recon[1,:], xlabel="Time index", ribbon=var_y_pred, label="Reconstructed x")
    plot!(1:n, y_recon[2,:], label="Reconstructed v", ribbon=var_y_pred, title="Model Prediction",    fillalpha = 10.0,
    fillcolor = :blue,)
    plot!(n+1:m, y_future[1,:], label=false)
    plot!(n+1:m, y_future[2,:], label=false, legend=:bottomleft)
end

begin
    figure2 = plot(2*(1:size(W,2)), W[1,1:end], xlabel="Time index", label="x")
    plot!(2*(1:size(W,2)), W[2,:], label="v", title="Observed timeseries", legend=:bottomleft)
    vline!([n], color=:red, linewidth=3, label="Training threshold")

    figure3 = plot(wl_features', label=false, title="WL features", xlabel="Time index")

    plot(figure2, figure1, layout=(3,1), size=(800, 1200))
    savefig("paper/timeseries.png")
end

savefig(figure2, "paper/ts-1.png")
savefig(figure1, "paper/ts-2.png")



err = y_pred[:,1:2:end-2] .- W[:,2:end]
metric = sqrt(mean(err.^2))


plot(y_pred[1,1:2:end])
plot!(W[1,2:end])
plot(y_pred[2,1:2:end])
plot!(W[2,2:end])