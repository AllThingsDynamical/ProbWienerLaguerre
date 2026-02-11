using BSplineKit
using Plots
using OrdinaryDiffEq
using FFTW
using NPZ
TEST = false

function interpolate(u::Matrix)
    d, M = size(u)
    interpolants = []
    t = LinRange(0.0, 1.0, M)
    for i=1:d
        interp = BSplineKit.interpolate(t, u[i,:], BSplineOrder(4))
        push!(interpolants, interp)
    end
    return interpolants
end

function divide_dataset(u::Matrix, window_size::Int)
    @assert window_size < size(u, 2)
    K = size(u,2) - 2*window_size
    inputs = []
    outputs = []
    for i=0:K-1
        inp = u[:,i+1:i+window_size]
        oup = u[:,i+window_size+1:i+2*window_size]
        push!(inputs, inp)
        push!(outputs, oup)
    end
    return inputs, outputs
end

function state_matrices(p::Int, λ::Float64)
    A = zeros(p, p)
    B = ones(p)
    for i=1:p
        for j=i:p
            if i==j
                A[i,i] = -λ
            else
                A[i,j] = -2*λ
            end
        end
    end
    B = sqrt(2*λ) * B
    return A, B
end

function wiener_laguerre_vector_field!(dv, v, p, t)
    A, B = state_matrices(p[3], p[2])
    ps = length(v)
    d = Int(ps/p[3])
    pd = p[3]
    u = p[1]
    for i=1:d
        dv[(i-1)*pd+1:i*pd] .= A*v[(i-1)*pd+1:i*pd] + B*(u[i](t))
    end
    nothing
end

function wiener_laguerre_features(p::Int, u::Any, resolution::Int, λ::Float64)
    d = length(u)
    ps = Int(p*d)
    w0 = zeros(ps)
    tspan = (0.0, 1.0)
    tsave = LinRange(0.0, 1.0, resolution)
    params = [u, λ, p]
    prob = ODEProblem(wiener_laguerre_vector_field!, w0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=tsave)
    Array(sol)
end

function evaluate_output_function(output_func::Any, resolution::Int)
    t = LinRange(0.0, 1.0, resolution)
    d = length(output_func)
    outputs = []
    for i=1:d
        push!(outputs, output_func[i].(t))
    end
    return Array(reduce(hcat, outputs)')
end

struct WL_Layer
    num_steps::Int
    resolution::Int
    order::Int
    decay_rate::Float64
end

function generate_input_output_features(u::Matrix, model::WL_Layer)
    k = model.num_steps
    resolution = model.resolution
    λ = model.decay_rate
    p = model.order

    inputs = u[:,1:end-k]
    outputs = u[:,k+1:end]
    input_func = interpolate(inputs)
    output_func = interpolate(outputs)

    wl_feature = wiener_laguerre_features(p, input_func, resolution, λ)
    output_feature = evaluate_output_function(output_func, resolution)

    return wl_feature, output_feature
end

function generate_sys_id_features(u::Matrix, y::Matrix, model::WL_Layer)
    k = model.num_steps
    resolution = model.resolution
    λ = model.decay_rate
    p = model.order

    input_func = interpolate(u)
    output_func = interpolate(y)

    wl_feature = wiener_laguerre_features(p, input_func, resolution, λ)
    output_feature = evaluate_output_function(output_func, resolution)

    return wl_feature, output_feature
end

function generate_input_features(u::Matrix, model::WL_Layer)
    M = model.resolution
    t = LinRange(0.0, 1.0, M)
    u_interp = interpolate(u)
    input_features = []
    d = size(u, 1)
    for i=1:d
        u_eval = u_interp[i].(t)
        push!(input_features, u_eval)
    end
    return reduce(hcat, input_features)
end

function munge_data(input_features::Matrix, output_features::Matrix; ratio=0.5)
    M = size(input_features, 2)
    N = Int(ratio*M)
    train_data = (input_features[:,1:N], output_features[:,1:N])
    test_data = (input_features[:,N+1:end], output_features[:,N+1:end])
    return train_data, test_data    
end

if TEST
    d = 1
    M = 5_000
    u = randn(d, M)
    k = 20
    p = 10
    resolution = 2*M
    λ = 0.5

    model = WL_Layer(k, resolution, p, λ)
    wl_features, output_features = generate_input_output_features(u, model)

    wl_features
    plot(wl_features')
end

if TEST
    d = 1
    M = 5_000
    u = randn(d, M)
    y = randn(1, M)
    k = 20
    p = 10
    resolution = 2*M
    λ = 0.5

    model = WL_Layer(k, resolution, p, λ)
    wl_features, output_features = generate_sys_id_features(u, y, model)

    figure1 = plot(wl_features')
    figure2 = plot(output_features')
end