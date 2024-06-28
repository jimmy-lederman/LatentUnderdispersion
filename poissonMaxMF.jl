include("matrixMF.jl")
include("poissonMaxFunctions.jl")
using Distributions

struct maxPoissonMF <: matrixMF
    N::Int
    M::Int
    K::Int
    a::Int
    b::Int
    c::Int
    d::Int
    D::Int
end

function evaluateLikelihod(model::maxPoissonMF, state, data, mask)
    Y_NM = data["Y_NM"]
    Mu_NM = state["U_NK"] * state["V_KM"]
    lik = pdf.(OrderStatistic.(Poisson.(Mu_NM), D, D), Y_NM)
    if !isnothing(mask)
        return lik .* mask
    else
        return lik
    end
end

function sample_prior(model::maxPoissonMF)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::maxPoissonMF, state=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), model.D, model.D))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::maxPoissonMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    P_K = zeros(model.K)
    Mu_NM = U_NK * V_KM
    # Loop over the non-zeros in Y_DV and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]), model.D, model.D))
                end
            end
            if Y_NM[n, m] > 0
                Z_NM[n, m] = sampleSumGivenMax(Y_NM[n, m], D, Poisson(Mu_NM[n, m]))
                P_K[:] = U_NK[n, :] .* V_KM[:, m]
                P_K[:] = P_K / sum(P_K)
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = 1/model.b + model.D*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = 1/model.d + model.D*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return data, state
end

N = 20
M = 20
K = 2
a = b = c = d = 1
D = 10
model = maxPoissonMF(N,M,K,a,b,c,d,D)
data, state = forward_sample(model)
posteriorsamples = fit(model, data, nsamples=100,nburnin=100,nthin=1)
print(evaluateInfoRate(model, data, posteriorsamples))
#fsamples, bsamples = gewekeTest(model, ["U_NK", "V_KM"], nsamples=1000, nburnin=100, nthin=1)