include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
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
    Dmax::Int64
    type::Int64 #1 is min, 2 is median, 3 is max
    p::Float64
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function update_D(model::OrderStatisticPoissonMF, Y_NM, mu_NM)
    if model.type == 2
        logprobs = [logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(model.p)/2 + (model.Dmax - d)*log(1-model.p)/2 for d in 1:2:model.Dmax]
    else
        logprobs = [logbinomial(model.Dmax-1, d-1) + (d-1)*log(model.p) + (model.Dmax - d)*log(1-model.p) for d in 1:model.Dmax]
    end
    @views for n in 1:model.N
        @views for m in 1:model.M
            Y = Y_NM[n,m]
            mu = mu_NM[n,m]
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
    end
    if model.type == 2
        D = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
    else
        D = argmax(rand(Gumbel(0,1), model.Dmax) .+ logprobs)
    end
    return D
end


function evalulateLogLikelihood(model::OrderStatisticPoissonMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    D = state["D"]
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end
    return logpmfOrderStatPoisson(Y,mu,D,j)
end

function sample_prior(model::OrderStatisticPoissonMF,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    if model.type == 2
        D = 2*rand(Binomial(Int((model.Dmax - 1)/2), model.p)) + 1
    else
        D = rand(Binomial(model.Dmax - 1, model.p)) + 1
    end


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D"=>D)
    return state
end

function forward_sample(model::OrderStatisticPoissonMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    D = state["D"]
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end
    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), D, j))
    Z_NMK = zeros(Int, model.N, model.M, model.K)
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

function backward_sample(model::OrderStatisticPoissonMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    D = copy(state["D"])
    if model.type == 1
        j = 1
    elseif model.type == 2
        j = div(D,2) + 1
    else
        j = D
    end

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
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]), D, j))
            end
        end
        # println(D)
        # println(j)
        # println(Poisson(Mu_NM[n, m]))
        # println(Y_NM[n, m])
        # println(sampleSumGivenOrderStatistic(Y_NM[n, m], D, j, Poisson(Mu_NM[n, m])))
        # println(Mu_NM[n, m])
        Z_NM[n, m] = sampleSumGivenOrderStatistic(Y_NM[n, m], D, j, Poisson(Mu_NM[n, m]))
        if Z_NM[n, m] > 0
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K / sum(P_K)))
        end
    end

    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + D*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + D*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    Mu_NM = U_NK * V_KM

    #last, infer D
    D = update_D(model, Y_NM, Mu_NM)

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D"=>D)
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