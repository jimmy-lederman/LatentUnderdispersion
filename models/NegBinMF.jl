include("matrixMF.jl")
using Distributions
using LinearAlgebra

struct NegBin <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
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

function evaluateLikelihod(model::NegBin, state, data, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    return pdf(NegativeBinomial(mu, model.p), Y)
end

function sample_prior(model::NegBin)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::NegBin; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Z_NMK = zeros(model.N, model.M, model.K)
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(NegativeBinomial.(Mu_NM, 1-model.p))
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    # for n in 1:model.N
    #     for m in 1:model.M
    #         for k in 1:model.K
    #             Z_NMK[n,m,k] = rand(Poisson(log(1/(1-model.p))*U_NK[n,k] * V_KM[k,m]))
    #         end
    #     end
    # end

    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK"=>U_NK, "V_KM"=>V_KM)
    return data, state 
end

function backward_sample(model::NegBin, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NMK = zeros(model.N, model.M, model.K)
    P_K = zeros(model.K)
    Z_NM = zeros(Int, model.N,model.M)
    
    Mu_NM = U_NK * V_KM
    #Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(NegativeBinomial(Mu_NM[n,m], 1-model.p))
                end
            end
            if Y_NM[n, m] > 0
                #sample CRT
                Z_NM[n,m] = sampleCRT(Y_NM[n,m],Mu_NM[n,m])
                #now Z is a (certain kind of) Poisson so we can thin it
                P_K[:] = U_NK[n, :] .* V_KM[:, m]
                P_K[:] = P_K / sum(P_K)
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + log(1/(1-model.p))*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + log(1/(1-model.p))*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
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