include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
using Distributions
using LinearAlgebra

struct MaxNegBinCovariateMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64
    h::Float64
    D::Int64
    p::Float64
end

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


function evalulateLogLikelihood(model::MaxNegBinCovariateMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    dist = info["dist_NM"][row,col]
    mu = dot(state["U_NK1"][row,:], state["V_KM1"][:,col]) + dist*dot(state["U_NK2"][row,:], state["V_KM2"][:,col]) 
    return  logpdf(OrderStatistic(NegativeBinomial(mu, 1-model.p), model.D, model.D), Y)
end


function sample_prior(model::MaxNegBinCovariateMF,info=nothing)
    U_NK1 = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM1 = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    U_NK2 = rand(Gamma(model.e, 1/model.f), model.N, model.K)
    V_KM2 = rand(Gamma(model.g, 1/model.h), model.K, model.M)
    state = Dict("U_NK1" => U_NK1, "V_KM1" => V_KM1,
                "U_NK2" => U_NK2, "V_KM2" => V_KM2, "dist_NM"=>info["dist_NM"])
    return state
end

function forward_sample(model::MaxNegBinCovariateMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
    end
    Mu_NM1 = state["U_NK1"] * state["V_KM1"]
    Mu_NM2 = state["U_NK2"] * state["V_KM2"]
    dist_NM = info["dist_NM"]
    Y_NM = rand.(OrderStatistic.(NegativeBinomial.(Mu_NM1 + Mu_NM2.*dist_NM, 1-model.p), model.D, model.D))
    # Z_NMK = zeros(Int, model.N, model.M, model.K)
    # for n in 1:model.N
    #     for m in 1:model.M
    #         for k in 1:model.K
    #             Z_NMK[n,m,k] = rand(Poisson(state["U_NK"][n,k]*state["V_KM"][k,m]))
    #         end
    #     end
    # end
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::MaxNegBinCovariateMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK1 = copy(state["U_NK1"])
    V_KM1 = copy(state["V_KM1"])
    U_NK2 = copy(state["U_NK2"])
    V_KM2 = copy(state["V_KM2"])
    dist_NM = copy(state["dist_NM"])
    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(model.N, model.M, 2*model.K)
    P_2K = zeros(2*model.K)
    
    Mu_NM1 = U_NK1 * V_KM1
    Mu_NM2 = U_NK2 * V_KM2
    #Loop over the non-zeros in Y_DV and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(Mu_NM1[n,m] + dist_NM[n,m]*Mu_NM2[n,m], 1-model.p), model.D, model.D))
                end
            end
            if Y_NM[n, m] > 0
                mu = Mu_NM1[n,m] + dist_NM[n,m]*Mu_NM2[n,m]
                temp = sampleSumGivenMax(Y_NM[n, m], model.D, NegativeBinomial(mu, 1-model.p))
                Z_NM[n,m] = sampleCRT(temp,model.D*mu)
                P_2K[1:model.K] = U_NK1[n, :] .* V_KM1[:, m]
                P_2K[(model.K+1):end] = dist_NM[n,m] * U_NK2[n, :] .* V_KM2[:, m]
                P_2K[:] = P_2K / sum(P_2K)
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_2K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + model.D*log(1/(1-model.p))*sum(V_KM1[k, :])
            U_NK1[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:model.K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + model.D*log(1/(1-model.p))*sum(U_NK1[:, k])
            V_KM1[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.e + sum(Z_NMK[n, :, model.K + k]) 
            post_rate = model.f + model.D*log(1/(1-model.p))*sum(V_KM2[k, :].*dist_NM[n,:])
            U_NK2[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:model.K
            post_shape = model.g + sum(Z_NMK[:, m, model.K + k]) 
            post_rate = model.h + model.D*log(1/(1-model.p))*sum(U_NK2[:, k].*dist_NM[:,m])
            V_KM2[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_NK1" => U_NK1, "V_KM1" => V_KM1, "U_NK2" => U_NK2, "V_KM2" => V_KM2, "dist_NM"=>dist_NM);
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