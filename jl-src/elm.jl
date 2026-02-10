using LinearAlgebra
using Plots

function sample_elm_weights(params::Tuple, d::Int)
    num_features = params[1]

    # optional scale (like a "lengthscale"); set params[2]=1.0 if you don't want it
    ℓ = params[2]
    W = randn(d, num_features) / ℓ

    # biases (you can also use rand(num_features) if you prefer)
    b = randn(num_features)

    return W, b
end

function elm_feature_matrix(X::Matrix{T}, params::Tuple) where {T}
    d, M = size(X)
    W, b = sample_elm_weights(params, d)
    K = params[1]

    # activation (swap to relu/tanh/sigmoid as needed)
    σ = z ->  tanh(z)

    ψ = x -> σ.(W' * x .+ b)

    Φ = zeros(M, K)
    for i = 1:M
        Φ[i, :] = ψ(X[:, i])
    end
    return Φ, W, b
end

function tikhonov_regularization(Φ::Matrix{T}, Y::Matrix{T}, params::Tuple) where {T}
    λ = params[3]
    # Φ is M×K, Y is n×M. Solve for a (K×n):
    # minimize ||Φ*a - Y'||_F^2 + λ||a||_F^2
    K = size(Φ, 2)
    a = pinv(Φ, atol=λ)*Y'
    return a
end

function construct_elm_estimator(a::Matrix{T}, W::Matrix{T}, b::AbstractVector{T}) where {T}
    k, n = size(a)

    σ = z -> tanh.(z)
    ψ = x -> σ.(W' * x .+ b)

    estimator = function(x::Matrix{T}) where {T}
        d, m = size(x)
        Φ = zeros(k, m)
        for i = 1:m
            Φ[:, i] = ψ(x[:, i])
        end
        return a' * Φ   # n×m
    end

    return estimator
end

function elm_estimator(X::Matrix{T}, Y::Matrix{T},
                       num_features::Int;
                       λ::T = T(1e-7)) where {T}
    d, _ = size(X)
    params = (num_features, d, λ)          # matches your skeleton
    Φ, W, b = elm_feature_matrix(X, params)
    a = tikhonov_regularization(Φ, Y, params)
    return construct_elm_estimator(a, W, b)
end

function bayesian_elm_estimator(X::Matrix{T}, Y::Matrix{T},
                                num_features::Int = 1200;
                                α::T = one(T),
                                σ2::T = T(1e-3)) where {T}

    d = size(X,1)
    params = (num_features, d, zero(T))
    Φ, W, b = rff_feature_matrix(X, params)

    M, K = size(Φ)
    n, My = size(Y)
    @assert My == M

    A = α * I(K) + (one(T)/σ2) * (Φ' * Φ)
    F = cholesky(Symmetric(Matrix(A)))

    m = (one(T)/σ2) * (F \ (Φ' * Y'))

    ψ = x -> cos.(W' * x .+ b)

    estimator = function(x::Matrix{T})
        Φx = ψ(x)
        μ = m' * Φx

        mtest = size(x, 2)
        v = zeros(T, mtest)
        for j = 1:mtest
            z = F.L \ Φx[:, j]
            v[j] = dot(z, z)
        end
        return μ, v
    end

    return estimator
end


if TEST
    d::Int = 10
    n::Int = 3
    M::Int = 10_00
    X = randn(d, M)
    Y = randn(n, M)

    estimator = elm_estimator(X, Y, 1200)
    y_pred = estimator(X)
    Y
    figure = plot(Y[:], Y[:], ms=0.1)
    scatter!(Y[:], y_pred[:], ms=0.1)
end

if TEST
    estimator = bayesian_elm_estimator(X, Y, 1200)
    mu_y_pred, var_y_pred = estimator(X)
    mu_y_pred
    Y
    figure = plot(mu_y_pred[1, :], ribbon=var_y_pred)
    display(figure)
    var_y_pred
end