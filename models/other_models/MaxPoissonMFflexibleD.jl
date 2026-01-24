include("MatrixMF.jl")
include("PoissonMaxFunctions.jl")
include("UpdateDFunctions.jl")
using Distributions
using LinearAlgebra
using LogExpFunctions
using PyCall
using SymPy

struct MaxPoissonMFflexibleD <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    r0::Float64
    p0::Float64
end


function evalulateLogLikelihood(model::MaxPoissonMFflexibleD, state, data, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    D = state["D_NM"][row,col]
    return logpmfMaxPoisson(Y,mu,D)
end

function sample_prior(model::MaxPoissonMFflexibleD)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    D_NM = rand(truncated(NegativeBinomial(model.r0,model.p0), lower=1), model.N, model.M)
    @assert all(x -> x >= 1, D_NM)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D_NM" => D_NM)
    return state
end

function forward_sample(model::MaxPoissonMFflexibleD; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    D_NM = state["D_NM"]
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    for n in 1:model.N
        for m in 1:model.M
            for k in 1:model.K
                Z_NMK[n,m,k] = rand(Poisson(state["U_NK"][n,k]*state["V_KM"][k,m]))
            end
        end
    end
    # Z_NM = dropdims(sum(Z_NMK, axis=2), dims=2)
    # Y_NM = maximum.()
    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), D_NM, D_NM))
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"], "D_NM" => D_NM, "Z_NMK"=>Z_NMK)
    return data, state 
end

function backward_sample(model::MaxPoissonMFflexibleD, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    D_NM = copy(state["D_NM"])
    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    P_K = zeros(model.K)
    Mu_NM = U_NK * V_KM

    #update each d
    for n in 1:model.N
        for m in 1:model.M
            D_NM[n,m] = sampleDposterior(model.r0,model.p0,Y_NM[n,m],Poisson(Mu_NM[n,m]))
        end
    end

    # #Loop over the non-zeros in Y_DV and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]), D_NM[n,m], D_NM[n,m]))
                end
            end
            if Y_NM[n, m] > 0
                if D_NM[n,m] == 1
                    Z_NM[n,m] = Y_NM[n, m]
                else
                    Z_NM[n, m] = sampleSumGivenMax(Y_NM[n, m], D_NM[n,m], Poisson(Mu_NM[n, m]))
                end
                P_K[:] = U_NK[n, :] .* V_KM[:, m]
                P_K[:] = P_K / sum(P_K)
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + dot(D_NM[n,:], V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:model.K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + dot(D_NM[:,m], U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D_NM" => D_NM, "Z_NMK" => Z_NMK)
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