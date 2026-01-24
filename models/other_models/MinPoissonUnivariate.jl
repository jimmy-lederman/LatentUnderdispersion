include("../helper/MatrixMF.jl")
include("../helper/PoissonMinFunctions.jl")
using Distributions

struct MinPoissonUnivariate <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    D::Int64
end

function evalulateLogLikelihood(model::MinPoissonUnivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    return logpmfMinPoisson(Y,mu,model.D)
end

function sample_prior(model::MinPoissonUnivariate, info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    state = Dict("mu" => mu)
    return state
end

function forward_sample(model::MinPoissonUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    Y_NM = rand(OrderStatistic(Poisson(mu), model.D, 1), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::MinPoissonUnivariate, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    Z_N = zeros(model.N)

    for n in 1:model.N
        Z_N[n] = sampleSumGivenMin(Y_NM[n,1], model.D, Poisson(mu))
    end

    post_shape = model.a + sum(Z_N)
    post_rate = model.b + model.D*model.N
    mu = rand(Gamma(post_shape, 1/post_rate))
    state = Dict("mu" => mu)
    return data, state
end
