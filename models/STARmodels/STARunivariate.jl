include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
using Distributions
using Base.Threads

struct STARunivariate <: MatrixMF
    N::Int64
    M::Int64
    mu0::Float64
    sigma0_2::Float64
    a::Float64
    b::Float64
    g::Function
    g_inv::Function
end

function STARlogpmf(x,mu,sigma2)
    return log(cdf(Normal(0,1), (model.g(x + .5) - mu)/sqrt(sigma2)) - cdf(Normal(0,1), (model.g(x - .5) - mu)/sqrt(sigma2)))
end


function evalulateLogLikelihood(model::STARunivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    sigma2 = state["sigma2"]
    return STARlogpmf(Y,mu,sigma2)
end

function sample_prior(model::STARunivariate, info=nothing)
    mu = rand(Normal(model.mu0, sqrt(model.sigma0_2)))
    sigma2 = rand(InverseGamma(model.a, model.b))
    @assert model.M == 1
    Z_N = rand(Normal(mu, sqrt(sigma2)), model.N)
    state = Dict("mu" => mu, "sigma2"=>sigma2, "Z_N"=>Z_N)
    return state
end

function forward_sample(model::STARunivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Z_N = state["Z_N"]
    @assert model.M == 1
    Y_NM = reshape(Int.(round.(model.g_inv.(Z_N))), :, 1)

    data = Dict("Y_NM" => copy(Y_NM))
    return data, state 
end

function backward_sample(model::STARunivariate, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    sigma2 = copy(state["sigma2"])
    Z_N = zeros(model.N)

    @views @threads for n in 1:model.N
        if Y_N[n] == 0
            Z_N[n] = rand(Truncated(Normal(mu, sqrt(sigma2)), -Inf, model.g(.5)))
        else
            Z_N[n] = rand(Truncated(Normal(mu, sqrt(sigma2)), model.g(Y_NM[n,1] - .5), model.g(Y_NM[n,1] + .5)))
        end
    end


    post_V = 1/(model.N/sigma2 + 1/model.sigma0_2)
    post_mu = post_V*(sum(Z_N)/sigma2 + model.mu0/model.sigma0_2)
    mu = rand(Normal(post_mu,sqrt(post_V)))

    post_a = model.a + model.N/2
    post_b = model.b + sum((Z_N .- mu).^2)/2
    sigma2 = rand(InverseGamma(post_a, post_b))

    state = Dict("mu" => mu, "sigma2"=>sigma2, "Z_N"=>Z_N)
    return data, state
end
