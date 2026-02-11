include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct covid2Poisson <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    c::Float64
    d::Float64
    g::Float64
    h::Float64
    scaleshape::Float64
    scalerate::Float64
end

function evalulateLogLikelihood(model::covid2Poisson, state, data, info, row, col)
    @assert !isnothing(info)
        if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]

    pop = info["pop_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+alpha*pop*mu+pop*eps
    return logpdf(Poisson(rate), Y)
end

function sample_prior(model::covid2Poisson, info=nothing,constantinit=nothing)
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    pop_sum = sum(info["pop_N"])
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, 
                "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"], "pop_sum"=>pop_sum)
    return state
end

function forward_sample(model::covid2Poisson; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Y0_N = state["Y0_N"]
    eps = state["eps"]
    pop_N = state["pop_N"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    

    Y_NM = zeros(Int, model.N, model.M)
    for m in 1:model.M
        for n in 1:model.N
            if m == 1
                Y_NM[n,m] = rand(Poisson(Y0_N[n] + pop_N[n]*eps + pop_N[n]*alpha*sum(U_NK[n,:] .* V_KM[:,m])))
            else
                Y_NM[n,m] = rand(Poisson(Y_NM[n,m-1] + pop_N[n]*eps + pop_N[n]*alpha*sum(U_NK[n,:] .* V_KM[:,m])))
            end
        end 
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid2Poisson, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Y0_N = copy(state["Y0_N"])
    pop_N = copy(state["pop_N"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])
    pop_sum = state["pop_sum"]
    #Y_NMKplus2 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    nt = Threads.nthreads()
    sum_factor_thr = zeros(Float64, nt)
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
        pop = pop_N[n]
        probvec = pop* U_NK[n,:] .* V_KM[:,m]
        mu1 = sum(probvec) 
        mu = Ylast + pop*eps + alpha*mu1
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(Poisson(mu))
            end
        end
        if Y_NM[n, m] > 0
            P_K = P_K_thr[tid]
            @inbounds begin
                @simd for k in 1:model.K
                    P_K[k] = alpha * probvec[k]
                end
                P_K[model.K+1] = Ylast
                P_K[model.K+2] = pop * eps
            end
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            @inbounds begin
                sum_factor_thr[tid] += Y_NM[n,m] - y_k[model.K+1] - y_k[model.K+2]
                sum_noise_thr[tid]  += y_k[model.K+2]
            end
            @inbounds for k in 1:model.K
                Y_NK_thr[tid][n, k] += y_k[k]
                Y_MK_thr[tid][m, k] += y_k[k]
            end
            #Y_NMKplus2[n, m, :] = y_k
        end
    end
    # try
    #     @assert sum(Y_NMKplus2) == sum(Y_NM)
    # catch ex
    #     println(sum(Y_NMKplus2))
    #     println(sum(Y_NM))
    #     @assert sum(Y_NMKplus2) == sum(Y_NM)
    # end
    sum_factor = sum(sum_factor_thr)
    sum_noise  = sum(sum_noise_thr)
    Y_NK  = sum(Y_NK_thr)  
    Y_MK  = sum(Y_MK_thr)  

    #update alpha (scale)
    post_shape = model.scaleshape + sum_factor
    sU = vec(sum(pop_N .* U_NK, dims=1))   # length K
    sV = vec(sum(V_KM, dims=2))            # length K
    post_rate = model.scalerate + dot(sU, sV)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    # #update noise term
    post_shape = model.g + sum_noise
    post_rate = model.h + model.M*pop_sum
    eps = rand(Gamma(post_shape, 1/post_rate))
    
    #Y_NK = dropdims(sum(Y_NMKplus2, dims = 2), dims = 2)
    A_K = fill(model.a, model.N)
    @views for k in 1:model.K
        U_NK[:, k] = rand(Dirichlet(A_K .+ Y_NK[:,k]))
    end

    @views for k in 1:model.K
        post_rate = model.d + alpha * sum(pop_N .* U_NK[:, k])
        @views for m in 1:model.M
            post_shape = model.c + Y_MK[m,k]
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"], "pop_sum"=>pop_sum)
    return data, state
end
