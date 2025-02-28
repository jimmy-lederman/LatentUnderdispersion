include("../helper/MatrixMF.jl")
include("../helper/OrderStatsSampling.jl")
using Distributions
using Base.Threads

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        return 1
    else
        probs = vcat([R/(R+i-1) for i in 2:Y])
    end
    return 1 + sum(rand.(Bernoulli.(probs)))
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

# function evalulateLogLikelihood(model::MaxNegBinUnivariate, state, data, info, row, col)
#     Y = data["Y_NM"][row,col]
#     mu = state["mu"]
#     return logpmfMaxPoisson(Y,mu,model.D)
# end

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

    Y_ND = rand(NegativeBinomial(mu, 1-p), model.N, model.D)
    Z1_N = sum(Y_ND,dims=2)
    #Y_NM = reshape(sort(Y_ND,dims=2)[:,model.j],model.N,1)

    Y_NM = rand(OrderStatistic(NegativeBinomial(mu, 1-p), model.D, model.j), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("mu"=>mu,"p"=>p,"Z1_N"=>Z1_N)
    return data, state 
end

function griddy_gibbs(model::OrderStatisticNegBinUnivariate, Z1sum, Z2sum, shape_factor,rate_factor,plist=.05:.001:.999)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    rlist = zeros(length(plist))
    logprobs = zeros(length(plist))
    for (i,p) in enumerate(plist)
        #for each p, sample an r from its complete conditional
        post_shape = model.a + shape_factor*Z2sum
        post_rate = model.b + rate_factor*log(1/(1-p))*model.N
        rlist[i] = rand(Gamma(post_shape, 1/post_rate))#post_shape/post_rate#
        #calculate logprob for griddy gibbs (r,p) pair
        logprobs[i] = logpdf(Beta(model.alpha,model.beta),p) + logpdf(NegativeBinomial(rate_factor*model.N*rlist[i], 1-p), Z1sum)
    end

    c = argmax(rand(Gumbel(0,1),length(logprobs)) .+ logprobs)
    return (plist[c], rlist[c])
end

function griddy_gibbs2(model::OrderStatisticNegBinUnivariate, Z1sum, Z2sum, plist=.01:.001:.99, rlist=1:2500)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    logprobs = zeros(length(plist),length(rlist))
    for (i,p) in enumerate(plist)
        for (j,r) in enumerate(rlist)
            #calculate logprob for griddy gibbs (r,p) pair
            logprobs[i,j] = logpdf(Gamma(model.a,1/model.b),r) + logpdf(Beta(model.alpha,model.beta),p) + logpdf(NegativeBinomial(model.D*model.N*r, 1-p), Z1sum)
        end
    end
    #println(logprobs)
    #probs = probs ./ sum(probs)
    c = argmax(rand(Gumbel(0,1),size(logprobs)) .+ logprobs)
    return (plist[c[1]], rlist[c[2]])
end


function backward_sample(model::OrderStatisticNegBinUnivariate, data, state, mask=nothing;griddy=false,annealStrat=nothing,anneal=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    p = copy(state["p"])
    Z1_N = zeros(model.N)
    #Z1_N = copy(state["Z1_N"])
    Z2_N = zeros(model.N)
    # println(mu)
    @views @threads for n in 1:model.N
        if isnothing(annealStrat) || annealStrat == 2
            Z1_N[n] = sampleSumGivenOrderStatistic2(Y_NM[n,1], model.D, model.j,NegativeBinomial(mu, 1-p))
            Z2_N[n] = sampleCRT(Z1_N[n], model.D*mu)
        else
            if annealStrat == 1
                Zlist = sampleListGivenOrderStatistic2(Y_NM[n,1], model.D, model.j,NegativeBinomial(mu, 1-p))
                Z1_N[n] = sum(Zlist[1:anneal])
                Z2_N[n] = sampleCRT(Z1_N[n],anneal*mu)
            elseif annealStrat == 3
                Z1_N[n] = sampleFirstKGivenOrderStatistic2(Y_NM[n,1], model.D, model.j,NegativeBinomial(mu, 1-p), anneal)
                #Z1_N[n] = sum(Zlist)
                Z2_N[n] = sampleCRT(Z1_N[n],anneal*mu)
            end
        end
    end
    if isnothing(annealStrat)
        rate_factor = model.D
        shape_factor = 1
    elseif annealStrat == 1 || annealStrat == 3
        rate_factor = anneal
        shape_factor = 1
    elseif annealStrat == 2 
        rate_factor = model.D/(model.D - anneal + 1)
        shape_factor = 1/(model.D - anneal + 1)
    end
    if griddy
        (p,mu) = griddy_gibbs(model, sum(Z1_N), sum(Z2_N), shape_factor, rate_factor)
    else
        post_alpha = model.alpha + shape_factor*sum(Z1_N)
        post_beta = model.beta + rate_factor*model.N*mu
        p = rand(Beta(post_alpha,post_beta))

        post_shape = model.a + shape_factor*sum(Z2_N)
        post_rate = model.b + rate_factor*log(1/(1-p))*model.N
        mu = rand(Gamma(post_shape, 1/post_rate))


       

    end




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