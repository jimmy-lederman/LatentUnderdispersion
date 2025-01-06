include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
include("../../helper/NegBinPMF.jl") 
using Distributions
using LinearAlgebra
using Base.Threads

struct genes <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
    dist::Function
end

function sampleCRT(Y,R)
    # if Y > 10000
    #     println(Y)
    # end
    if Y == 0
        return 0
    elseif Y == 1
        probs = [1]
    else
        probs = vcat([1],[R/(R+i-1) for i in 2:Y])
    end
    return sum(rand.(Bernoulli.(probs)))
end


function evalulateLogLikelihood(model::genes, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    if isnothing(state["p_N"])
        if model.D == 1
            return logpdf(model.dist(mu), Y)
        else
            #return logpdf(OrderStatistic(mode.dist(mu), model.D, model.j), Y)
            if model.D == model.j
                return logpmfMaxPoisson(Y, mu, model.D)
            else
                return logpdf(OrderStatistic(mode.dist(mu), model.D, model.j), Y)
            end
        end
    else
        p = state["p_N"][row]
        if model.D == 1
            return logpdf(model.dist(mu,1-p), Y)
        else
            if model.D == model.j
                return logpmfMaxNegBin(Y, mu, p, model.D)
            else
                return logpdf(OrderStatistic(model.dist(mu,1-p), model.D, model.j), Y)
            end
        end
    end
    return logpmfMaxPoisson(Y,mu,model.D)
end

function sample_prior(model::genes,info=nothing)
    p_N = nothing
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    try
        dist = model.dist(1)
    catch e 
        p_N = rand(Beta(model.alpha, model.beta), model.N)
    end
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p_N" => p_N)
end

function sample_likelihood(model::genes, mu,p=nothing)
    if isnothing(p)
        dist = model.dist(mu)
    else
        dist = model.dist(mu,1-p)
    end
    if model.D == 1
        return rand(dist)
    else
        return rand(OrderStatistic(dist, model.D, model.j))
    end
end

function forward_sample(model::genes; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    # Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = zeros(Int, model.N, model.M)
    #Z_NMK = zeros(Int, model.N, model.M, model.K)
    p_N = state["p_N"]
    for n in 1:model.N
        if !isnothing(p_N)
            p = p_N[n]
        else
            p=nothing
        end
        for m in 1:model.M
            Y_NM[n,m] = sample_likelihood(model,sum(state["U_NK"][n,:].*state["V_KM"][:,m]),p)

        end
    end
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"], "p_N"=>p_N)
    return data, state 
end

function backward_sample(model::genes, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    p_N = state["p_N"]
    if !isnothing(p_N)
        p_N = copy(p_N)
    end
    Z1_NM = zeros(Int, model.N, model.M)
    Z2_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    Mu_NM = U_NK * V_KM
    #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        mu = Mu_NM[n,m]
        if isnothing(p_N)
            p = nothing
            lik = model.dist
        else
            p = p_N[n]
            lik = x -> model.dist(x,1-p)
        end
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = sample_likelihood(model, mu, p)
            end
        end
        if Y_NM[n, m] > 0 || model.j != model.D

            Z1_NM[n, m] = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, model.j, lik(mu))
            if !isnothing(p_N)
                Z2_NM[n,m] = copy(Z1_NM[n,m])
                Z1_NM[n,m] = sampleCRT(Z1_NM[n,m], model.D*mu)
            end
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z1_NM[n, m], P_K / sum(P_K)))
        end
    end
    
    if !isnothing(p_N)
        post_alpha = model.alpha .+ sum(Z2_NM, dims=2)
        post_beta = model.beta .+ model.D .* sum(Mu_NM, dims=2)
        p_N = rand.(Beta.(post_alpha,post_beta))
        rate_factor_N = model.D*log.(1 ./(1 .- p_N))
    else
        p2 = nothing
        rate_factor_N = fill(model.D, model.N)
    end



    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + rate_factor_N[n]*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + sum(rate_factor_N .* U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p_N"=>p_N)
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