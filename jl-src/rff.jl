using LinearAlgebra
using Plots

```The objective is to provide an implementation of the random Fourier features algorithm
that approximates the Gram matrix of a Gaussian process or kernel function using a squared exponential covariance function.
```

d::Int = 10
n::Int = 3
M::Int = 10_00
X = randn(d, M)
Y = randn(n, M)

function sample_rff_weights(params::Tuple, d::Int)
    num_features = params[1]
    
    b = 2*π*rand(num_features)

    l = 1/params[2]
    W = l*randn(d, num_features)
    
    return W, b
end

function feature_matrix(X::Matrix{T}, params::Tuple) where {T}
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

function construct_estimator(a::Matrix{T}, W::Matrix{T}, b::Any) where {T}
    d, M = size(X)
    k, n = size(a)
    ψ = x -> cos.(W'*x + b)
    
    estimator = function(x::Matrix{T}) where {T}
        d, m = size(x)
        Φ = zeros(k, m)
        for i=1:m
            Φ[:,i] = ψ(x[:,i])
        end
        return a'*Φ
    end

    return estimator
end

num_features = 1200
ρ = 1.0
λ = 1e-7
params = (num_features, ρ, λ)
Φ, W, b = feature_matrix(X, params)
a = tikhonov_regularization(Φ, Y, params)
estimator = construct_estimator(a, W, b)

y_pred = estimator(X)
Y

figure = plot(Y[:], Y[:], ms=0.1)
scatter!(Y[:], y_pred[:], ms=0.1)