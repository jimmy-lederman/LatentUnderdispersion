include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
using Distributions
using Base.Threads

struct Gaussian <: MatrixMF
    N::Int64
    M::Int64
    mu0::Float64
    sigma0_2::Float64
    a::Float64
    b::Float64
end



function evalulateLogLikelihood(model::Gaussian, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    sigma2 = state["sigma2"]
    return logpdf(normal(mu, sqrt(sigma2)))
end

function sample_prior(model::Gaussian, info=nothing)
    mu = rand(Normal(model.mu0, sqrt(model.sigma0_2)))
    sigma2 = rand(InverseGamma(model.a, model.b))
    @assert model.M == 1
    state = Dict("mu" => mu, "sigma2"=>sigma2)
    return state
end

function forward_sample(model::Gaussian; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    @assert model.M == 1
    Y_NM = zeros(model.N, model.M)
    mu = copy(state["mu"])
    sigma2 = copy(state["sigma2"])
    Y_NM[:,1] = rand(Normal(mu, sqrt(sigma2)), model.N)


    data = Dict("Y_NM" => copy(Y_NM))
    return data, state 
end

function backward_sample(model::Gaussian, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    sigma2 = copy(state["sigma2"])

    post_V = 1/(model.N/sigma2 + 1/model.sigma0_2)
    post_mu = post_V*(sum(Y_NM)/sigma2 + model.mu0/model.sigma0_2)
    mu = rand(Normal(post_mu,sqrt(post_V)))

    post_a = model.a + model.N/2
    post_b = model.b + sum((Y_NM .- mu).^2)/2
    sigma2 = rand(InverseGamma(post_a, post_b))

    state = Dict("mu" => mu, "sigma2"=>sigma2)
    return data, state
end
