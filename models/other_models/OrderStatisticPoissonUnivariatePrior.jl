include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads
using SpecialFunctions

struct OrderStatisticPoissonUnivariatePrior <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    Dmax::Int64
    type::Int64
    p::Float64
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function update_D(model::OrderStatisticPoissonUnivariatePrior, Y_NM, mu)
    if model.type == 2
        logprobs = [logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(model.p)/2 + (model.Dmax - d)*log(1-model.p)/2 for d in 1:2:model.Dmax]
    else
        logprobs = [logbinomial(model.Dmax-1, d-1) + (d-1)*log(model.p) + (model.Dmax - d)*log(1-model.p) for d in 1:model.Dmax]
    end
    @views for n in 1:model.N
        Y = Y_NM[n,1]
        @views for d in 1:model.Dmax
            if model.type == 2 && d % 2 == 0
                continue
            end
            if model.type == 1
                j = 1
                logprobs[d] += logpmfOrderStatPoisson(Y, mu, d, j)
            elseif model.type == 2
                j = div(d,2) + 1
                logprobs[Int(div(d,2)+1)] += logpmfOrderStatPoisson(Y, mu, d, j)
            else
                j = d
                logprobs[d] += logpmfOrderStatPoisson(Y, mu, d, j)
            end
        end
    end

    if model.type == 2
        D = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
    else
        D = argmax(rand(Gumbel(0,1), model.Dmax) .+ logprobs)
    end
    return D
end

function evalulateLogLikelihood(model::OrderStatisticPoissonUnivariatePrior, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    D = state["D"]
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end
    return logpdf(OrderStatistic(Poisson(mu),D,j),Y)
end

function sample_prior(model::OrderStatisticPoissonUnivariatePrior, info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    if model.type == 2
        D = 2*rand(Binomial(Int((model.Dmax - 1)/2), model.p)) + 1
    else
        D = rand(Binomial(model.Dmax - 1, model.p)) + 1
    end
    state = Dict("mu" => mu, "D"=>D)
    return state
end

function griddy_gibbs(model::OrderStatisticPoissonUnivariatePrior, Y_NM,  mu)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    if type == 2
        Dlist = 1:2:model.Dmax
    else
        Dlist = 1:model.Dmax
    end
    mulist = zeros(length(Dlist))
    if model.type == 2
        logprobs = [logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(model.p)/2 + (model.Dmax - d)*log(1-model.p)/2 for d in 1:2:model.Dmax]
    else
        logprobs = [logbinomial(model.Dmax-1, d-1) + (d-1)*log(model.p) + (model.Dmax - d)*log(1-model.p) for d in 1:model.Dmax]
    end
    for (i,D) in enumerate(Dlist)
        Z_N = zeros(model.N)
        if model.type == 1
            j = 1
        elseif model.type == 2
            j = div(D,2) + 1
        else
            j = D
        end
        @views @threads for n in 1:model.N
            Z_N[n] = sampleSumGivenOrderStatistic(Y_NM[n,1], D, j, Poisson(mu))
        end
        
        #for each p, sample an r from its complete conditional
       #update mu
        post_shape = model.a + sum(Z_N)
        post_rate = model.b + D*model.N
        mulist[i] = rand(Gamma(post_shape, 1/post_rate))#post_shape/post_rate#
        #calculate logprob for griddy gibbs (mu,D) pair
        logprobs[i] += sum(logpdf.(Poisson(D*mulist[i]), Z_N))
    end
    println(logprobs)
    @assert 1 == 2

    c = argmax(rand(Gumbel(0,1),length(logprobs)) .+ logprobs)
    return (Dlist[c], mulist[c])
end

function forward_sample(model::OrderStatisticPoissonUnivariatePrior; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    D = state["D"]
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end
    @assert model.M == 1

    Y_NM = rand(OrderStatistic(Poisson(mu), D, j),model.N,model.M)
    data = Dict("Y_NM" => copy(Y_NM))
    #state = Dict("mu"=>mu)
    return data, state 
end

function backward_sample(model::OrderStatisticPoissonUnivariatePrior, data, state, mask=nothing;skipupdate=nothing,griddy=false)
    #some housekeeping
    Y_NM = data["Y_NM"]
    mu = copy(state["mu"])
    D = copy(state["D"])
    
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end

    if griddy
        (D,mu) = griddy_gibbs(model, Y_NM, mu)
    else
        Z_N = zeros(model.N)
        @views @threads for n in 1:model.N
            Z_N[n] = sampleSumGivenOrderStatistic(Y_NM[n,1], D, j, Poisson(mu))
        end

        #update mu
        post_shape = model.a + sum(Z_N)
        post_rate = model.b + D*model.N
        mu = rand(Gamma(post_shape, 1/post_rate))

        #update D
        if isnothing(skipupdate) || !("D" in skipupdate)
            D = update_D(model, Y_NM, mu)
        end
    end
    

    state = Dict("mu" => mu, "D"=>D)
    return data, state
end
