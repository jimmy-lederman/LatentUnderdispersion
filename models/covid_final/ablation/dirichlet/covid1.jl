include("../../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct covid1 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::covid1, state, data, info, row, col)
        @assert !isnothing(info)
    if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]
    mu = Ylast + dot(state["U_NK"][row,:], state["V_KM"][:,col])
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::covid1,info=nothing,constantinit=nothing)
    U_NK = rand(Dirichlet(ones(model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid1; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y0_N = state["Y0_N"]
    Y_NM = zeros(Int, model.N, model.M)
    for m in 1:model.M
        for n in 1:model.N
            if m == 1
                Y_NM[n,m] = rand(Poisson(Y0_N[n] + Mu_NM[n,m]))
            else
                Y_NM[n,m] = rand(Poisson(Y_NM[n,m-1] + Mu_NM[n,m]))
            end
        end 
    end

    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid1, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Y0_N = copy(state["Y0_N"])
    nt = Threads.nthreads()
    # Loop over the non-zeros in Y_NM and allocate
    Y_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Y_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K + 1) for _ in 1:nt]

    @views @threads for idx in 1:(model.N * model.M)
        tid = Threads.threadid()
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        if !isnothing(mask)
            if mask[n,m] == 1
                P_K = P_K_thr[tid]
                @inbounds begin
                    @simd for k in 1:model.K
                        P_K[k] =  U_NK[n,k] * V_KM[k,m]
                    end
                    P_K[model.K+1] = Ylast
                end
                Y_NM[n,m] = rand(Poisson(sum(P_K)))
            end
        end
        if Y_NM[n, m] > 0
            if mask[n,m] == 0
                P_K = P_K_thr[tid]
                @inbounds begin
                    @simd for k in 1:model.K
                        P_K[k] = U_NK[n,k] * V_KM[k,m]
                    end
                    P_K[model.K+1] = Ylast
                end
            end
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            @inbounds for k in 1:model.K
                Y_NK_thr[tid][n, k] += y_k[k]
                Y_MK_thr[tid][m, k] += y_k[k]
            end
        end
    end
    #@assert sum(Y_NMK) == sum(Y_NM)
    Y_NK  = sum(Y_NK_thr)  
    Y_MK  = sum(Y_MK_thr) 
    
    #Y_NK = dropdims(sum(Y_NMK, dims = 2), dims = 2)
    @views for k in 1:model.K
         U_NK[:, k] = rand(Dirichlet(ones(model.N) .+ Y_NK[:,k]))
    end
    
    @views for k in 1:model.K
        post_rate = model.d + sum(U_NK[:, k])
        @views for m in 1:model.M
            post_shape = model.c + Y_MK[m,k]
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y0_N"=>info["Y0_N"])
    return data, state
end
