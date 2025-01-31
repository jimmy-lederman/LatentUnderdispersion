include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct PoissonMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::PoissonMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonMF,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::PoissonMF, state=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(Poisson.(Mu_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::PoissonMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Y_NMK = zeros(model.N, model.M, model.K)
    if !isnothing(mask)
        Mu_NM = U_NK * V_KM
    end
    # Loop over the non-zeros in Y_NM and allocate
    #locker = Threads.SpinLock()
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1    
        if !isnothing(mask)
            if mask[n,m] == 1
                lock(locker)
                Y_NM[n,m] = rand(Poisson(Mu_NM[n,m]))
                unlock(locker)
            end
        end
        if Y_NM[n, m] > 0
            P_K = U_NK[n, :] .* V_KM[:, m]
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            Y_NMK[n, m, :] = y_k
        end
    end
    @assert sum(Y_NMK) == sum(Y_NM)
    
    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Y_NMK[n, :, k])
            post_rate = model.b + sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Y_NMK[:, m, k])
            post_rate = model.d + sum(U_NK[:, k])
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