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
    λ = 20.0

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

    figure0 = plot(idx1, input_features, label="Train Input")
    plot!(idx2, input_features_t, label="Test Input", xlabel="#Iter")

    figure1 = plot(idx1, wl_features', label=false, title="WL features")
    plot!(idx2, wl_features_t', label=false, xlabel="#Iter")

    figure2 = plot(idx1, output_features', label="Train Output")
    plot!(idx2, output_features_t', label="Test Output", title="Reference", xlabel="#Iter") 
end

begin
    idx_shift = 1
    X = wl_features[:, idx_shift:end]
    Y = output_features[:, idx_shift:end]
    estimator = rff_estimator(X, Y, 200, λ=1e-1)

    mu_y_pred = estimator(X)
    k = 1
    W = hcat(X[:,end-k:end], wl_features_t)
    mu_y_test = estimator(W)

    idxk = resolution1-k:resolution1 + resolution2
    figure3 = plot(idx_shift:idx1[end], mu_y_pred', label="Reconstructed",
                    ylims=[-0.48, 0.48])
    plot!(idxk, mu_y_test', label="Test Output", title="Prediction", xlabel="#Iter") 

    plot(figure0, figure1, figure2, figure3, size=(1200, 600))
end