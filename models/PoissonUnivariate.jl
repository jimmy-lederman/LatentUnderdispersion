include("../helper/MatrixMF.jl")
using Distributions

struct PoissonUnivariate <: MatrixMF
    N::Int64 #number of data points
    M::Int64 #always set to 1
    a::Float64 
    b::Float64
end

function evalulateLogLikelihood(model::PoissonUnivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonUnivariate, info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    state = Dict("mu" => mu)
    return state
end

function forward_sample(model::PoissonUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    Y_NM = rand(Poisson(mu), model.N)
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::PoissonUnivariate, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])

    post_shape = model.a + sum(Y_NM)
    post_rate = model.b + model.N
    mu = rand(Gamma(post_shape, 1/post_rate))
    state = Dict("mu" => mu)
    return data, state
end

