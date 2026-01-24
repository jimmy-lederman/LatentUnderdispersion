include("../helper/MatrixMF.jl")
include("../helper/PoissonMedianFunctions.jl")
using Distributions

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        probs = [1]
    else
        probs = vcat([1],[R/(R+i-1) for i in 2:Y])
    end
    return sum(rand.(Bernoulli.(probs)))
end

struct MedianNegBinUnivariate <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    alpha::Float64
    beta::Float64
    D::Int64
end

# function evalulateLogLikelihood(model::MaxNegBinUnivariate, state, data, info, row, col)
#     Y = data["Y_NM"][row,col]
#     mu = state["mu"]
#     return logpmfMaxPoisson(Y,mu,model.D)
# end

function sample_prior(model::MedianNegBinUnivariate, info=nothing,constantinit=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    p = rand(Beta(model.alpha, model.beta))
    state = Dict("mu" => mu, "p" => p)
    return state
end

function forward_sample(model::MedianNegBinUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    p = state["p"]

    Y_ND = rand(NegativeBinomial(mu, 1-p), model.N, model.D)
    Z1_N = sum(Y_ND,dims=2)
    Y_NM = reshape(median(Y_ND,dims=2),model.N,1)

    #Y_NM = rand(OrderStatistic(NegativeBinomial(mu, 1-p), model.D, div(model.D,2)+1), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("mu"=>mu,"p"=>p,"Z1_N"=>Z1_N)
    return data, state 
end

function backward_sample(model::MedianNegBinUnivariate, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    p = copy(state["p"])
    Z1_N = zeros(model.N)
    #Z1_N = copy(state["Z1_N"])
    Z2_N = zeros(model.N)

    for n in 1:model.N
        @assert model.D == 3
        Z1_N[n] = sampleSumGivenMedian3(Y_NM[n,1], NegativeBinomial(mu, 1-p))
        Z2_N[n] = sampleCRT(Z1_N[n], model.D*mu)
    end

    post_alpha = model.alpha + sum(Z1_N)
    post_beta = model.beta + model.D*model.N*mu
    p = rand(Beta(post_alpha,post_beta))


    post_shape = model.a + sum(Z2_N)
    post_rate = model.b + model.D*log(1/(1-p))*model.N
    mu = rand(Gamma(post_shape, 1/post_rate))


    state = Dict("mu" => mu, "p" => p)
    return data, state
end

# N = 100
# M = 100
# K = 2
# a = b = c = d = 1
# D = 10
# model = maxPoissonMF(N,M,K,a,b,c,d,D)
# data, state = forward_sample(model)
# posteriorsamples = fit(model, data, nsamples=100,nburnin=100,nthin=1)
# print(evaluateInfoRate(model, data, posteriorsamples))
#fsamples, bsamples = gewekeTest(model, ["U_NK", "V_KM"], nsamples=1000, nburnin=100, nthin=1)