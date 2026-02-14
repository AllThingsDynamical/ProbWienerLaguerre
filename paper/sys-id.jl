using NPZ
using Plots
include("../jl-src/custom_plot.jl")
include("../jl-src/wl-model.jl")
include("../jl-src/estimators.jl")

begin
    filename = "data/strum_l.npz"
    data = npzread(filename)
    U_train = reshape(data["U_train"], 1, :)
    Y_train = reshape(data["Y_train"], 1, :)
    U_test = reshape(data["U_test"], 1, :)
    Y_test = reshape(data["Y_test"], 1, :)
end

begin
    k = 5
    p = 15
    λ = 30.0

    M1 = size(U_train, 2)
    resolution1 = 2*M1
    M2 = size(U_test, 2)
    resolution2 = 2*M2

    model1 = WL_Layer(k, resolution1, p, λ) 
    wl_features, output_features = generate_sys_id_features(U_train, Y_train, model1)
    input_features = generate_input_features(U_train, model1)

    model2 = WL_Layer(k, resolution2, p, λ) 
    wl_features_t, output_features_t = generate_sys_id_features(U_test, Y_test, model2)
    input_features_t = generate_input_features(U_test, model2)
end

begin
    idx1 = 1:resolution1
    idx2 = resolution1+1:resolution1+resolution2

    figure0 = plot(idx1, input_features, label="Train", title="Input signal")
    plot!(idx2, input_features_t, label="Test", xlabel="Time index")

    figure1 = plot(idx1, wl_features', label=false, title="WL features")
    plot!(idx2, wl_features_t', label=false, xlabel="Time index")

    figure2 = plot(idx1, output_features', label="Train")
    plot!(idx2, output_features_t', label="Test", title="True Output signal", xlabel="Time index") 
end

begin
    idx_shift = 1
    X = wl_features[:, idx_shift:end]
    Y = output_features[:, idx_shift:end]
    estimator = bayesian_rff_estimator(X, Y, 1000, σ2=8e-1)

    mu_y_pred, var_y_pred = estimator(X)
    k = -1
    W = hcat(X[:,end-k:end], wl_features_t)
    mu_y_test, var_y_test = estimator(W)

    idxk = resolution1-k:resolution1 + resolution2
    figure3 = plot(idx_shift:idx1[end], mu_y_pred', label="Reconstructed",
                    ylims=[-0.48, 0.48], ribbon=var_y_pred[:])
    plot!(idxk, mu_y_test', label="Test", title="Model Prediction", xlabel="Time index",
            ribbon=var_y_test) 

    plot(figure0, figure2, figure3, layout=(4,1), size=(800, 1500))
    savefig("paper/sys-id.png")
end

savefig(figure0, "paper/sys-id-1.png")
savefig(figure2, "paper/sys-id-2.png")
savefig(figure3, "paper/sys-id-3.png")

using Statistics
rmse = x -> sqrt(mean(x[:].^2))
err = mu_y_test[:,1:2:end] .- Y_test
metric = rmse(err)

mean(var_y_test[1:2:end])