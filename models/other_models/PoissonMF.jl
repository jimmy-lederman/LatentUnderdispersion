include("../../helper/MatrixMF.jl")
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

function sample_prior(model::PoissonMF,info=nothing,constantinit=nothing)
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::PoissonMF; state=nothing,info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
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
    #Y_NMK = zeros(model.N, model.M, model.K)
    if !isnothing(mask)
        Mu_NM = U_NK * V_KM
    end
    nt = Threads.nthreads()
    Y_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Y_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K + 2) for _ in 1:nt]
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(model.N * model.M)
        tid = Threads.threadid()
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(Poisson(Mu_NM[n,m]))
            end
        end
        if Y_NM[n, m] > 0
            P_K = P_K_thr[tid]
            @inbounds for k in 1:model.K
                P_K[k] = U_NK[n, k] * V_KM[k, m]
            end
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            @inbounds for k in 1:model.K
                Y_NK_thr[tid][n, k] += y_k[k]
                Y_MK_thr[tid][m, k] += y_k[k]
            end
        end
    end
    Y_NK  = sum(Y_NK_thr)  
    Y_MK  = sum(Y_MK_thr)  
    #@assert sum(Y_NMK) == sum(Y_NM)
    A_K = fill(model.a, model.N)
    @views for k in 1:model.K
        U_NK[:, k] = rand(Dirichlet(A_K .+ Y_NK[:,k]))
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + Y_MK[m,k]
            post_rate = model.d + 1
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
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