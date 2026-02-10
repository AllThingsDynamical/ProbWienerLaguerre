using NPZ
using Statistics
include("../jl-src/estimators.jl")

function estimate_trimodal_gaussian_with_elm(k::Int)
    filename = "data/gaussian_$k.npz"
    file = npzread(filename)

    X, Y = file["X_train"], file["Y_train"]
    Xt, Yt = file["X_test"], file["Y_test"]

    @time begin
        BELM_estimator = bayesian_elm_estimator(X, Y, 1_500, σ2 = 1e-6)

        y_test = BELM_estimator(X)
        figure1 = plot(Y[:], Y[:])
        scatter!(Y[:], y_test[1][:], ms=0.1)

        y_pred = BELM_estimator(Xt)
        figure2 = plot(Yt[:], Yt[:])
        scatter!(Yt[:], y_pred[1][:], ms=0.1)
        display(plot(figure1, figure2))
    end

    err_pred = (y_pred[1] .- Yt) ./ 1
    mean_err_pred = mean(abs.(err_pred))
    mean_std_err_pred = sqrt.(mean(y_pred[2]))

    @info "ELM, d = $k"
    @show mean_err_pred
    @show mean_std_err_pred
    print("=====================================") 
end

K = [1,2,3,4,5]
for k in K
    estimate_trimodal_gaussian_with_elm(k)
end


function estimate_trimodal_gaussian_with_rff(k::Int)
    filename = "data/gaussian_$k.npz"
    file = npzread(filename)

    X, Y = file["X_train"], file["Y_train"]
    Xt, Yt = file["X_test"], file["Y_test"]

    @time begin
        BELM_estimator = bayesian_rff_estimator(X, Y, 1_500, ρ = 2.0, σ2 = 1e-6)

        y_test = BELM_estimator(X)
        figure1 = plot(Y[:], Y[:])
        scatter!(Y[:], y_test[1][:], ms=0.1)

        y_pred = BELM_estimator(Xt)
        figure2 = plot(Yt[:], Yt[:])
        scatter!(Yt[:], y_pred[1][:], ms=0.1)
        display(plot(figure1, figure2))
    end

    err_pred = (y_pred[1] .- Yt) ./ 1
    mean_err_pred = mean(abs.(err_pred))
    mean_std_err_pred = sqrt.(mean(y_pred[2]))

    @info "RFF, d = $k"
    @show mean_err_pred
    @show mean_std_err_pred
    print("=====================================") 
end

K = [1,2,3,4,5]
for k in K
    estimate_trimodal_gaussian_with_rff(k)
end