include("../helper/MatrixMF.jl")
include("../helper/OrderStatsSampling.jl")
include("../helper/PoissonOrderPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct OrderStatisticPoissonMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    D::Int64
    j::Int64
end


function evalulateLogLikelihood(model::OrderStatisticPoissonMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    return logpdf(OrderStatistic(Poisson(mu), model.D, model.j), Y)
end

function sample_prior(model::OrderStatisticPoissonMF,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::OrderStatisticPoissonMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), model.D, model.j))
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    # for n in 1:model.N
    #     for m in 1:model.M
    #         for k in 1:model.K
    #             Z_NMK[n,m,k] = rand(Poisson(state["U_NK"][n,k]*state["V_KM"][k,m]))
    #         end
    #     end
    # end
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"])
    return data, state 
end

function backward_sample(model::OrderStatisticPoissonMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    Mu_NM = U_NK * V_KM
    #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1    
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]), model.D, model.j))
            end
        end
        Z_NM[n, m] = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, model.j, Poisson(Mu_NM[n, m]))
        if Z_NM[n, m] > 0
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K / sum(P_K)))
        end
    end

    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + model.D*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + model.D*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
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