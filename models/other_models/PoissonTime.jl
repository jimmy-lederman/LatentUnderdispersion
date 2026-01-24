include("../helper/MatrixMF.jl")
using Distributions

struct PoissonTime <: MatrixMF
    N::Int64 #number of data points
    M::Int64 #always set to 1
    a::Float64 
    b::Float64
end

function evalulateLogLikelihood(model::PoissonTime, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    Ylast = data["Y_NM"][row,col]
    mu = state["mu"]
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonTime, info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    state = Dict("mu" => mu)
    return state
end

function forward_sample(model::PoissonTime; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = copy(state["mu"])
    Y_NMold = copy(info["Y_NM"])
    Y_NMnew = copy(info["Y_NM"])
    for n in model.N
        if n == 1 continue end
        Y_NMnew[n,1] = rand(Poisson(mu*Y_NMold[n-1,1]))
    end 
    data = Dict("Y_NM" => Y_NMnew)
    state = Dict("mu"=>mu,"Y_NMold"=>Y_NMold)
    return data, state 
end

function backward_sample(model::PoissonTime, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    Y_NMold = copy(state["Y_NMold"])

    post_shape = model.a + sum(Y_NM[2:model.N,1])
    post_rate = model.b + sum(Y_NMold[1:(model.N-1),1])
    mu = rand(Gamma(post_shape, 1/post_rate))
    state = Dict("mu" => mu,"Y_NMold"=>Y_NMold)
    return data, state
end
