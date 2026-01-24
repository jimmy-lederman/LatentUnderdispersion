include("../../helper/MatrixMF.jl")
include("../../helper/NegBinPMF.jl")
using Distributions
using LinearAlgebra

struct NegBinMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
end

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

function evalulateLogLikelihood(model::NegBinMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    p = state["p"]
    return logpdf(NegativeBinomial(mu, 1-p), Y)
end

function sample_prior(model::NegBinMF)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    p = rand(Beta(model.alpha,model.beta))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p"=>p)
    return state
end

function forward_sample(model::NegBinMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Z_NMK = zeros(model.N, model.M, model.K)
    Mu_NM = state["U_NK"] * state["V_KM"]
    p = state["p"]
    p_NM = fill(1-p,model.N, model.M)
    Y_NM = rand.(NegativeBinomial.(Mu_NM, p_NM))
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    p = state["p"]

    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::NegBinMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    p = state["p"]
    Z_NMK = zeros(model.N, model.M, model.K)
    P_K = zeros(model.K)
    Z_NM = zeros(Int, model.N,model.M)
    
    Mu_NM = U_NK * V_KM
    #Loop over the non-zeros in Y_NM and allocate
    @views @threads for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(NegativeBinomial(Mu_NM[n,m], 1-p))
                end
            end
            if Y_NM[n, m] > 0
                #sample CRT
                Z_NM[n,m] = sampleCRT(Y_NM[n,m],Mu_NM[n,m])
                #now Z is a (certain kind of) Poisson so we can thin it
                P_K = U_NK[n, :] .* V_KM[:, m]
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K / sum(P_K)))
            end
        end
    end

    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + log(1/(1-p))*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for m in 1:model.M
        @views for k in 1:K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + log(1/(1-p))*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    Mu_NM = U_NK * V_KM

    p = rand(Beta(model.alpha + sum(Y_NM), model.beta + sum(Mu_NM)))


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p"=>p)
    return data, state
end

# N = 100
# M = 100
# K = 10
# a = b = c = d = 1
# model = poissonMF(N,M,K,a,b,c,d)
# # data, state = forward_sample(model)
# # posteriorsamples = fit(model, data)
# fsamples, bsamples = gewekeTest(model, ["U_NK", "V_KM"], nsamples=1000, nburnin=100, nthin=1)