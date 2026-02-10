using LinearAlgebra
using Plots

```The objective is to provide an implementation of the random Fourier features algorithm
that approximates the Gram matrix of a Gaussian process or kernel function using a squared exponential covariance function.
```

function sample_rff_weights(params::Tuple, d::Int)
    num_features = params[1]
    
    b = 2*π*rand(num_features)

    l = 1/params[2]
    W = l*randn(d, num_features)
    
    return W, b
end

function rff_feature_matrix(X::Matrix{T}, params::Tuple) where {T}
    d, M = size(X)
    W, b = sample_rff_weights(params, d)
    K = params[1]
    ψ = x -> cos.(W'*x + b)
    Φ = zeros(M, K)
    for i=1:M
        Φ[i,:] = ψ(X[:,i])
    end
    return Φ, W, b
end

function tikhonov_regularization(Φ::Matrix{T}, Y::Matrix{T}, params::Tuple) where {T}
    λ = params[3]
    a = pinv(Φ, atol=λ)*Y'
    return a
end

function construct_rff_estimator(a::Matrix{T}, W::Matrix{T}, b::Any) where {T}
    d, M = size(X)
    k, n = size(a)
    ψ = x -> cos.(W'*x + b)
    
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

function rff_estimator(X::Matrix{T}, Y::Matrix{T},
                       num_features::Int;
                       ρ::T = one(T),
                       λ::T = T(1e-7)) where {T}
    params = (num_features, ρ, λ)        # (K, rho, lambda)
    Φ, W, b = rff_feature_matrix(X, params)
    a = tikhonov_regularization(Φ, Y, params)
    return construct_rff_estimator(a, W, b)
end

function bayesian_rff_estimator(X::Matrix{T}, Y::Matrix{T},
                                num_features::Int = 1200;
                                ρ::T = one(T),
                                α::T = one(T),
                                σ2::T = T(1e-3)) where {T}

    params = (num_features, ρ, zero(T))
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

begin
    d::Int = 10
    n::Int = 3
    M::Int = 10_00
    X = randn(d, M)
    Y = randn(n, M)
end

begin
    estimator = rff_estimator(X, Y, 1200)
    y_pred = estimator(X)
    Y
    figure = plot(Y[:], Y[:], ms=0.1)
    scatter!(Y[:], y_pred[:], ms=0.1)
end

begin
    estimator = bayesian_rff_estimator(X, Y, 1200)
    mu_y_pred, var_y_pred = estimator(X)
    mu_y_pred
    Y
    figure = plot(mu_y_pred[1, :], ribbon=var_y_pred)
    display(figure)
    var_y_pred
end