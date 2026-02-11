include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
using Distributions
using Base.Threads

struct STARunivariate2a <: MatrixMF
    N::Int64
    M::Int64
    mu0::Float64
    sigma0_2::Float64
    a::Float64
    b::Float64
end

# function STARlogpmf(x,mu,sigma2)
#     return log(cdf(Normal(0,1), (model.g(x + .5) - mu)/sqrt(sigma2)) - cdf(Normal(0,1), (model.g(x - .5) - mu)/sqrt(sigma2)))
# end


# function evalulateLogLikelihood(model::STARunivariate2a, state, data, info, row, col)
#     Y = data["Y_NM"][row,col]
#     mu = state["mu"]
#     sigma2 = state["sigma2"]
#     return STARlogpmf(Y,mu,sigma2)
# end

function sample_prior(model::STARunivariate2a, info=nothing)
    mu = rand(Normal(model.mu0, sqrt(model.sigma0_2)))
    sigma2 = rand(InverseGamma(model.a, model.b))
    @assert model.M == 1
    state = Dict("mu" => mu, "sigma2"=>sigma2)
    return state
end

function forward_sample(model::STARunivariate2a; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = copy(state["mu"])
    sigma2 = copy(state["sigma2"])
    Z_N = rand(Normal(mu, sqrt(sigma2)), model.N)
    @assert model.M == 1
    Y_N = zeros(model.N)
    @views for n in 1:model.N
        if Z_N[n] < 0
            Y_N[n] = 0
        else
            Y_N[n] = ceil(Z_N[n])
        end
    end

    data = Dict("Y_N" => copy(Y_N))
    return data, state 
end

function backward_sample(model::STARunivariate2a, data, state, mask=nothing)
    #some housekeeping
    Y_N = copy(data["Y_N"])
    # println(Y_N)
    # @assert 1 == 2
    mu = copy(state["mu"])
    sigma2 = copy(state["sigma2"])
    #Z_N = copy(state["Z_N"])
    Z_N = zeros(model.N)
    for n in 1:model.N
        if Y_N[n] == 0
            Z_N[n] = rand(Truncated(Normal(mu, sqrt(sigma2)), -Inf, 0))
        else
            Z_N[n] = rand(Truncated(Normal(mu, sqrt(sigma2)), Y_N[n] - 1, Y_N[n]))         
            @assert Z_N[n] > 0
        end
    end
    

    post_V = 1/(model.N/sigma2 + 1/model.sigma0_2)
    post_mu = post_V*(sum(Z_N)/sigma2 + model.mu0/model.sigma0_2)
    mu = rand(Normal(post_mu,sqrt(post_V)))

    post_a = model.a + model.N/2
    post_b = model.b + sum((Z_N .- mu).^2)/2
    sigma2 = rand(InverseGamma(post_a, post_b))

    state = Dict("mu" => mu, "sigma2"=>sigma2)
    return data, state
end
