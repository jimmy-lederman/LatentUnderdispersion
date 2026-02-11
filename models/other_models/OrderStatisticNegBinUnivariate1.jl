include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
#include("../../helper/OrderStatsSampling_old.jl")
using Distributions
using Base.Threads

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end

struct OrderStatisticNegBinUnivariate <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
end

function evalulateLogLikelihood(model::OrderStatisticNegBinUnivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    p = state["p"]
    return logpmfOrderStatNegBin(Y, mu, p, model.D, model.j)
end

function sample_prior(model::OrderStatisticNegBinUnivariate, info=nothing,constantinit=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    p = rand(Beta(model.alpha, model.beta))
    state = Dict("mu" => mu, "p" => p)
    return state
end

function forward_sample(model::OrderStatisticNegBinUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    p = state["p"]


    Y_NM = rand(OrderStatistic(NegativeBinomial(mu, p), model.D, model.j), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("mu"=>mu,"p"=>p)
    return data, state 
end

function griddy_gibbs(model::OrderStatisticNegBinUnivariate, Z1vec, Z2sum,plist=.01:.01:.99)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    rlist = zeros(length(plist))
    logprobs = zeros(length(plist))
    for (i,p) in enumerate(plist)
        #for each p, sample an r from its complete conditional
        post_shape = model.a + Z2sum
        post_rate = model.b + model.D*log(1/p)*model.N
        rlist[i] = rand(Gamma(post_shape, 1/post_rate))#post_shape/post_rate#
        #calculate logprob for griddy gibbs (r,p) pair
        logprobs[i] = logpdf(Beta(model.alpha,model.beta),p) + sum(logpdf.(NegativeBinomial(model.D*rlist[i], p), Z1vec))
    end

    c = argmax(rand(Gumbel(0,1),length(logprobs)) .+ logprobs)
    return (plist[c], rlist[c])
end


function backward_sample(model::OrderStatisticNegBinUnivariate, data, state, mask=nothing;annealStrat=nothing,anneal=nothing,griddy=false)
    #some housekeeping
    Y_NM = data["Y_NM"]
    mu = copy(state["mu"])
    p = copy(state["p"])
    
    nt = Threads.nthreads()
    z1sum_nt = [0 for _ in 1:nt]
    z2sum_nt = [0 for _ in 1:nt]
    

    if griddy
        Z1_N = zeros(model.N)
        @views @threads for n in 1:model.N
            tid = Threads.threadid()
            z1 = sampleSumGivenOrderStatistic(Y_NM[n,1], model.D, model.j, NegativeBinomial(mu, p))
            Z1_N[n] = z1
            z1sum_nt[tid] += z1
            z2 = sampleCRT(z1, model.D*mu)
            z2sum_nt[tid] += z2
        end
        z1sum = sum(z1sum_nt)
        z2sum = sum(z2sum_nt)
        (p,mu) = griddy_gibbs(model, Z1_N, z2sum)
        # println(p)
    else 
        @views @threads for n in 1:model.N
            tid = Threads.threadid()
            z1 = sampleSumGivenOrderStatistic(Y_NM[n,1], model.D, model.j, NegativeBinomial(mu, p))
            z1sum_nt[tid] += z1
            z2 = sampleCRT(z1, model.D*mu)
            z2sum_nt[tid] += z2
        end
        z1sum = sum(z1sum_nt)
        z2sum = sum(z2sum_nt)
        post_shape = model.a + z2sum
        post_rate = model.b + log(1/p)*model.N*model.D
        mu = rand(Gamma(post_shape, 1/post_rate))
        p = rand(Beta(model.alpha + mu*model.N*model.D, model.beta + z1sum))
    end

    state = Dict("mu" => mu, "p" => p)
    return data, state
end
