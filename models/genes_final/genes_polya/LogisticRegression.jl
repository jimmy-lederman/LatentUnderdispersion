include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
#include("../../helper/NegBinPMF.jl") 
using Distributions
using LinearAlgebra
using Base.Threads
#include("PolyaGammaHybridSamplers.jl/src/pghybrid.jl")
using PolyaGammaHybridSamplers

using LogExpFunctions

struct LogisticRegression <: MatrixMF
    N::Int64
    P::Int64
    m_P::Vector
    B_PP::Matrix
    
end


function sample_prior(model::LogisticRegression,info=nothing)
    #mu = rand(Gamma(model.a, 1/model.b))
    Beta_P = rand(MvNormal(model.m_P,model.B_PP))
    X_NP = info["X_NP"]
    p_N = logistic.(X_NP * Beta_P)

    
    state = Dict("p_N" => p_N, "Beta_P"=>Beta_P)
end

function forward_sample(model::LogisticRegression; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Y_N = zeros(Int, model.N)
    p_N = state["p_N"]
    for n in 1:model.N
        Y_N[n] = rand(Bernoulli(p_N[n]))
    end
    data = Dict("Y_N" => Y_N)
    return data, state
end

function backward_sample(model::LogisticRegression, data, state, mask=nothing)
    #some housekeeping
    Y_N = copy(data["Y_N"])
    Beta_P = copy(state["Beta_P"])
    p_N = copy(state["p_N"])
    #println("beep")
    #flush(stdout)

    #Polya-gamma augmentation to update Beta_P
    W_N = zeros(Float64, model.N)
    @views for n in 1:model.N
        #sample a polya-gamma random variable
        #pg = PolyaGammaPSWSampler(1, sum(Beta_P .* X_NP[n,:]))
        pg = PolyaGammaHybridSampler(1, sum(Beta_P .* X_NP[n,:]))
        W_N[n] = rand(pg)
    end
    #println(W_N)
    #update Beta_P
    W = diagm(W_N)
    #println(W)
    V = inv(X_NP' * W * X_NP .+ inv(model.B_PP))
    #println(V)
    V = .5(V + V')
    k = Y_N .- .5
    
    mvec = V*(X_NP' * k .+ inv(model.B_PP)*model.m_P)
    #println(k)
    #println(mvec, " ", V)
    Beta_P = rand(MvNormal(mvec,V))

    p_N = logistic.(X_NP * Beta_P)

    state = Dict("p_N" => p_N, "Beta_P"=>Beta_P)
    return data, state
end