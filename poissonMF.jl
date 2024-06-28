include("matrixMF.jl")
using Distributions

struct poissonMF <: matrixMF
    N::Int
    M::Int
    K::Int
    a::Int
    b::Int
    c::Int
    d::Int
end

function evaluateLikelihod(model::poissonMF, state, data, mask=nothing)
    Y_NM = data["Y_NM"]
    Mu_NM = state["U_NK"] * state["V_KM"]
    lik = pdf.(Poisson.(Mu_NM), Y_NM)
    if !isnothing(mask)
        return lik .* mask
    else
        return lik
    end
end

function sample_prior(model::poissonMF)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::poissonMF, state=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(Poisson.(Mu_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::poissonMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Y_NMK = zeros(model.N, model.M, model.K)
    P_K = zeros(model.K)
    if !isnothing(mask)
        Mu_NM = U_NK * V_KM
    end
    # Loop over the non-zeros in Y_DV and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(Poisson(Mu_NM[n,m]))
                end
            end
            if Y_NM[n, m] > 0
                P_K[:] = U_NK[n, :] .* V_KM[:, m]
                P_K[:] = P_K / sum(P_K)
                Y_NMK[n, m, :] = rand(Multinomial(Y_NM[n, m], P_K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Y_NMK[n, :, k])
            post_rate = 1/model.b + sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:K
            post_shape = model.c + sum(Y_NMK[:, m, k])
            post_rate = 1/model.d + sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return data, state
end

N = 100
M = 100
K = 10
a = b = c = d = 1
model = poissonMF(N,M,K,a,b,c,d)
# data, state = forward_sample(model)
# posteriorsamples = fit(model, data)
fsamples, bsamples = gewekeTest(model, ["U_NK", "V_KM"], nsamples=1000, nburnin=100, nthin=1)