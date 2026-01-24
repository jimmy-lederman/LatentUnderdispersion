include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
#include("../../helper/NegBinPMF.jl") 
using Distributions
using LinearAlgebra
using Base.Threads
include("PolyaGammaHybridSamplers.jl/src/pghybrid.jl")

using LogExpFunctions

struct NegBinRegression <: MatrixMF
    N::Int64
    P::Int64
    a::Float64
    b::Float64
    m_P::Vector
    B_PP::Matrix
    
end

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        return 1
    else
        probs = [R/(R+i-1) for i in 2:Y]
    end
    return 1 + sum(rand.(Bernoulli.(probs)))
end

function sample_prior(model::NegBinRegression,info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    Beta_P = rand(MvNormal(model.m_P,model.B_PP))
    X_NP = info["X_NP"]
    p_N = logistic.(X_NP * Beta_P)

    
    state = Dict("mu" => mu, "p_N" => p_N, "Beta_P"=>Beta_P)
end

function sample_likelihood(model::NegBinRegression, mu,p=nothing)
    return rand(NegativeBinomial(mu,p))
end

function forward_sample(model::NegBinRegression; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Y_N = zeros(Int, model.N)
    p_N = state["p_N"]
    for n in 1:model.N
        Y_N[n] = sample_likelihood(model,sum(state["mu"]),p_N[n])
    end
    data = Dict("Y_N" => Y_N)
    return data, state
end

function backward_sample(model::NegBinRegression, data, state, mask=nothing)
    #some housekeeping
    Y_N = copy(data["Y_N"])
    mu = copy(state["mu"])
    Beta_P = copy(state["Beta_P"])
    p_N = copy(state["p_N"])

    Z_N = zeros(Int, model.N)

    @views for n in 1:model.N
        Z_N = sampleCRT(Y_N[n], mu)
    end

    #Polya-gamma augmentation to update p,
    W_N = zeros(Float64, model.N)
    @views for n in 1:model.N
        #sample a polya-gamma random variable
        pg = PolyaGammaHybridSampler(Y_N[n] + mu, sum(X_NP[n,:] .* Beta_P))
        W_N[n] = rand(pg)
    end

    #update Beta_P
    W = diagm(W_N)
    V = inv(X_NP' * W * X_NP .+ inv(model.B_PP))
    V = .5(V + V')
    k = (.5*(mu .- Y_N))
    
    mvec = V*(X_NP' * k .+ inv(model.B_PP)*model.m_P)
    
    Beta_P = rand(MvNormal(mvec,V))

    p_N = logistic.(X_NP * Beta_P)

    # post_shape = model.a + sum(Y_N)
    # post_rate = model.b + sum(log.(1 ./(1 .- p_N)))
    # #println(post_shape, " ", post_rate)
    # mu = rand(Gamma(post_shape, 1/post_rate))

    state = Dict("mu" => mu, "p_N" => p_N, "Beta_P"=>Beta_P)
    return data, state
end