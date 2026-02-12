include("../../../../helper/MatrixMF.jl")
include("../../../../helper/OrderStatsSampling.jl")
include("../../../../helper/PoissonOrderPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads
using SpecialFunctions

struct covid3 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    D::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    g::Float64
    h::Float64
    # i::Float64
    # j::Float64
    start_V1::Float64
    start_V2::Float64
end

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end

function evalulateLogLikelihood(model::covid3, state, data, info, row, col)
    @assert !isnothing(info)
        if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]

    eps = state["eps"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+mu+eps
    if model.D == 1
        return logpdf(Poisson(rate), Y)
    else
        return logpmfOrderStatPoisson(Y,rate,model.D,div(model.D,2)+1)
    end
end

function sample_prior(model::covid3, info=nothing,constantinit=nothing)
    pass = false
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.start_V1,1/model.start_V2))
                    #V_KM[k,m] = rand(Gamma(model.c*model.start_V+model.d,1/model.c))
                else
                    V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1]+model.d,1/model.c))
                end
            end
        end
    end
    #lambda_K = rand(Gamma(model.i, 1/model.j), model.K)
    eps = rand(Gamma(model.g, 1/model.h))
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, #"lambda_K" =>lambda_K,
                "eps"=>eps, 
    "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid3; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Y0_N = state["Y0_N"]
    eps = state["eps"]
    U_NK = state["U_NK"]
    #lambda_K = state["lambda_K"]
    V_KM = state["V_KM"]

    Y_NM = zeros(Int, model.N, model.M)
    for m in 1:model.M
        for n in 1:model.N
            if m == 1
                #Y_NM[n,m] = rand(Poisson(Y0_N[n] + pop_N[n]*eps + pop_N[n]*sum(U_NK[n,:] .* V_KM[:,m] .* lambda_K)))
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y0_N[n] + eps + sum(U_NK[n,:] .* V_KM[:,m] )), model.D, div(model.D, 2) + 1))
            else
                #Y_NM[n,m] = rand(Poisson(Y_NM[n,m-1] + pop_N[n]*eps + pop_N[n]*sum(U_NK[n,:] .* V_KM[:,m] .* lambda_K)))
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + eps + sum(U_NK[n,:] .* V_KM[:,m])), model.D, div(model.D, 2) + 1))
            end
        end 
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid3, data, state, mask=nothing;skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    #lambda_K = copy(state["lambda_K"])
    Y0_N = copy(state["Y0_N"])
    eps = copy(state["eps"])

    #Y_NMKplus2 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    nt = Threads.nthreads()
    sum_noise_thr  = zeros(Float64, nt)
    Y_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Y_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K + 2) for _ in 1:nt]
    
    @views @threads for idx in 1:(model.N * model.M)
        tid = Threads.threadid()
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        P_K = P_K_thr[tid]
        @inbounds begin
            @simd for k in 1:model.K
                P_K[k] =  U_NK[n,k] * V_KM[k,m]
            end
            P_K[model.K+1] = Ylast
            P_K[model.K+2] = eps
        end
        mu = sum(P_K) 
        if !isnothing(mask)
            if mask[n,m] == 1   
                Y_NM[n,m] = rand(OrderStatistic(Poisson(mu), model.D, div(model.D, 2) + 1))
            end
        end
        Z = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, div(model.D,2)+1, Poisson(mu))
        if Z > 0
            y_k = rand(Multinomial(Z,  P_K / sum(P_K)))
            @inbounds begin
                sum_noise_thr[tid]  += y_k[model.K+2]
            end
            @inbounds for k in 1:model.K
                Y_NK_thr[tid][n, k] += y_k[k]
                Y_MK_thr[tid][m, k] += y_k[k]
            end
        end
    end

    sum_noise  = sum(sum_noise_thr)
    Y_NK  = sum(Y_NK_thr)  
    Y_MK  = sum(Y_MK_thr)  

    #update noise term
    post_shape = model.g + sum_noise
    post_rate = model.h + model.D*model.M*model.N
    eps = rand(Gamma(post_shape, 1/post_rate))
    
    @views for k in 1:model.K
        post_rate = model.a + model.D*sum(V_KM[k, :])
        @views for n in 1:model.N
            post_shape = model.b + Y_NK[n,k]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    C_K = model.D*dropdims(sum(U_NK, dims=1),dims=1)
    if !isnothing(skipupdate) && ("V_KM" in skipupdate)
        #do regular update during part of burn-in
        
        @views for m in 1:model.M
            @views for k in 1:model.K
                post_shape = model.c + Y_MK[m,k]
                post_rate = model.d + C1_K[k]
                V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
            end
        end
    else
        l_KM = zeros(model.K,model.M+1)
        q_KM = zeros(model.K,model.M+1)
        
        #backward pass
        @views @threads for k in 1:model.K
            @views for m in model.M:-1:2
                q_KM[k,m] = log(1 + (C_K[k]/model.c) + q_KM[k,m+1])
                
                temp = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
                
                l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
            end 
        end
        
        @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
        #forward pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.start_V1 + Y_MK[m,k] + l_KM[k,m+1], 1/(model.start_V2 + C_K[k] + model.c*q_KM[k,m+1])))
                else
                    V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C_K[k] + model.c*q_KM[k,m+1])))
                end 
            end 
        end 
    end


    # @views for k in 1:model.K
    #     shape_post = model.i + sum(Y_MK[:,k])
    #     rate_post  = model.j + C_K[k] * sum(V_KM[k,:])
    #     lambda_K[k] = rand(Gamma(shape_post, 1 / rate_post))
    # end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps"=>eps,# "lambda_K"=>lambda_K,
    "Y0_N"=>info["Y0_N"])
    return data, state
end
