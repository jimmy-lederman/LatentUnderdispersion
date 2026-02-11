include("../../../helper/MatrixMF.jl")
include("../../../helper/OrderStatsSampling.jl")
include("../../../helper/PoissonOrderPMF.jl")
include("../../genes_final/genes_polya/PolyaGammaHybridSamplers.jl/src/pghybrid.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
using SpecialFunctions
using Base.Threads

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end


struct covid4 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    Q::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    g::Float64 
    h::Float64
    start_V1::Float64
    start_V2::Float64
    alpha0::Float64
    beta0::Float64
    tauc::Float64
    taud::Float64
    start_tau::Float64
end


function evalulateLogLikelihood(model::covid4, state, data, info, row, col)
    @assert !isnothing(info)
    if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]
    D = state["D_NM"][row,col]
    eps = state["eps"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+mu+eps
    if D == 1
        return logpdf(Poisson(rate), Y)
    else
        return logpmfOrderStatPoisson(Y,rate,D,div(D,2)+1)
    end
end

function sample_likelihood(model::covid4, mu,D,n=1)
    if D == 1
        return rand(Poisson(mu),n)
    else
        return rand(OrderStatistic(Poisson(mu),D, div(D,2)+1),n)
    end
end

function sample_prior(model::covid4,info=nothing,constantinit=nothing)
    pass = false
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    #U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    #V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
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
    eps = rand(Gamma(model.g, 1/model.h))

    #sigma_N = rand(InverseGamma(model.alpha0, model.beta0), model.N)
    #sigma_M = rand(InverseGamma(model.alpha0, model.beta0), model.M)
    sigma2_beta = rand(InverseGamma(model.alpha0, model.beta0))
    sigma2_tau = rand(InverseGamma(model.alpha0, model.beta0))

    Beta_NQ = rand(Normal(0,sqrt(sigma2_beta)),model.N,model.Q)
    Tau_QM = zeros(model.Q, model.M)
    @views for m in 1:model.M
        @views for q in 1:model.Q
            if m == 1
                Tau_QM[q,m] = rand(Normal(model.tauc*model.start_tau+model.taud,sqrt(sigma2_tau)))
            else
                Tau_QM[q,m] = rand(Normal(model.tauc*Tau_QM[q,m-1]+model.taud,sqrt(sigma2_tau)))
            end
        end
    end
    D_NM = 2*rand.(Binomial.(Int((model.Dmax - 1)/2), logistic.(Beta_NQ * Tau_QM))) .+ 1


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, #"R_KTS" => R_KTS, 
    "eps"=>eps, "D_NM"=>D_NM,
    "sigma2_beta"=>sigma2_beta, "sigma2_tau"=>sigma2_tau, "Beta_NQ"=>Beta_NQ, "Tau_QM"=>Tau_QM,
     "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid4; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    eps = state["eps"]
    Y0_N = state["Y0_N"]

    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    D_NM = state["D_NM"]

    Y_NM = zeros(Int, model.N, model.M)

    for m in 1:model.M
        for n in 1:model.N
            D = D_NM[n,m]
            if m == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y0_N[n] + eps + sum(U_NK[n,:] .* V_KM[:,m])), D, div(D,2)+1))
            else
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + eps + sum(U_NK[n,:] .* V_KM[:,m])),D, div(D,2)+1))
            end
        end 
    end

    data = Dict("Y_NM" => Y_NM)

    return data, state 
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function update_D2(model::covid4, Y, mu, p)
    Dlist = 1:2:model.Dmax
    jlist = div.(Dlist,2) .+ 1
    logpmfs = logpmfOrderStatPoissonVec(Y,mu,Dlist,jlist,compute=false)
    logpriors = [logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(p)/2 + (model.Dmax - d)*log(1-p)/2 for d in 1:2:model.Dmax]
    D = 2*argmax(rand(Gumbel(0,1), length(logpmfs)) .+ logpmfs .* logpriors) - 1
    @assert D >= 1 && D <= model.Dmax
    return D
end

function update_D(model::covid4, Y, mu, p)
    logprobs = [logpmfOrderStatPoisson(Y,mu,d,div(d,2)+1,compute=false) + logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(p)/2 + (model.Dmax - d)*log(1-p)/2 for d in 1:2:model.Dmax]
    D = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
    @assert D >= 1 && D <= model.Dmax
    return D
end

function backward_sample(model::covid4, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_NQ = copy(state["Beta_NQ"])
    Tau_QM = copy(state["Tau_QM"])
    D_NM = copy(state["D_NM"])
    sigma2_beta = copy(state["sigma2_beta"])
    sigma2_tau = copy(state["sigma2_tau"])

    eps = copy(state["eps"])

    Y0_N = copy(state["Y0_N"])
    p_NM = logistic.(Beta_NQ * Tau_QM)

    # Loop over the non-zeros in Y_NM and allocate
    nt = Threads.nthreads()
    sum_noise_thr  = zeros(Float64, nt)
    Y_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Y_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K + 2) for _ in 1:nt]
    @views @threads for idx in 1:(model.N * model.M)
        tid = Threads.threadid()
        m = div(idx - 1, model.N) + 1
        n = mod(idx - 1, model.N) + 1
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
                Y_NM[n,m] = rand(OrderStatistic(Poisson(mu), D_NM[n,m], div(D_NM[n,m], 2) + 1))
            end
        end
        Z = sampleSumGivenOrderStatistic(Y_NM[n, m], D_NM[n,m], div(D_NM[n,m],2)+1, Poisson(mu))
        if Z > 0 
            y_k = rand(Multinomial(Z, P_K / sum(P_K)))
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

    #update county factors
    A_K = fill(model.a, model.N)
    @views for k in 1:model.K
        U_NK[:, k] = rand(Dirichlet(A_K .+ Y_NK[:,k]))
    end

    #update noise term
    post_shape = model.g + sum_noise
    post_rate = model.h + sum(D_NM)
    eps = rand(Gamma(post_shape, 1/post_rate))

    # #update time factors
    # C1_KM = U_NK' * D_NM

    # l_KM = zeros(model.K,model.M+1)
    # q_KM = zeros(model.K,model.M+1)
    
    # #backward pass
    # @views for m in model.M:-1:2
    #     @views for k in 1:model.K
    #         q_KM[k,m] = log(1 + (C1_KM[k,m]/model.c) + q_KM[k,m+1])
            
    #         temp = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
            
    #         l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
    #     end 
    # end
    
    # @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    # #forward pass
    # @views for m in 1:model.M
    #     @views for k in 1:model.K
    #         if m == 1
    #             V_KM[k,m] = rand(Gamma(model.start_V1 + Y_MK[m,k] + l_KM[k,m+1], 1/(model.start_V2 + C1_KM[k,m] + model.c*q_KM[k,m+1])))
    #         else
    #             V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C1_KM[k,m] + model.c*q_KM[k,m+1])))
    #         end 
    #     end 
    # end 
    
    # #Polya-gamma augmentation to update D, Beta and Tau
    # if isnothing(skipupdate) || !("D_NM" in skipupdate)
    #     F_NM = Beta_NQ * Tau_QM
    #     W_NM = zeros(Float64, model.N, model.M)
    #     p_NM = logistic.(F_NM)
    #     @views @threads for idx in 1:(model.N * model.M)
    #         m = div(idx - 1, model.N) + 1
    #         n = mod(idx - 1, model.N) + 1
    #         if m == 1
    #             Ylast = Y0_N[n]
    #         else
    #             Ylast = Y_NM[n,m-1]
    #         end
    #         mu = Ylast + eps + sum(U_NK[n,:] .* V_KM[:,m]) 

    #         D_NM[n,m] = update_D(model, Y_NM[n,m], mu, p_NM[n,m])
    #         pg = PolyaGammaHybridSampler((model.Dmax - 1)/2, F_NM[n,m])
    #         W_NM[n,m] = rand(pg)
    #     end


    #     Ident = Matrix{Int}(I, model.Q, model.Q)
    #     @views @threads for m in 1:model.M
    #         W = Diagonal(W_NM[:,m])
    #         if m == M
    #             V = inv(Beta_NQ' * W * Beta_NQ + (1/sigma2_tau)*Ident)
    #         else
    #             V = inv(Beta_NQ' * W * Beta_NQ + (1/sigma2_tau + model.tauc^2 / sigma2_tau)*Ident)
    #         end
    #         V = .5(V + V')
    #         k = (D_NM[:,m] .- 1)/2 .- (model.Dmax - 1)/4
    #         if m == 1
    #             mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*Ident*(model.taud .+ model.tauc*fill(model.start_tau,model.Q)) + (model.tauc/sigma2_tau)*Ident*(Tau_QM[:,m+1] .- model.taud)))
    #         elseif m == M
    #             mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*Ident*(model.taud .+ model.tauc*Tau_QM[:,m-1])))
    #         else
    #             mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*Ident*(model.taud .+ model.tauc*Tau_QM[:,m-1]) + (model.tauc/sigma2_tau)*Ident*(Tau_QM[:,m+1] .- model.taud)))
    #         end
    #         Tau_QM[:,m] = rand(MvNormal(mvec,V))
    #         #update variance
    #         # if m == 1
    #         #     sigma_M[m] = rand(InverseGamma(model.alpha0 + model.Q/2, model.beta0 + sum((Tau_QM[:,m] .- model.taud .- model.tauc*fill(model.start_tau,model.Q)).^2)/2))
    #         # else
    #         #     sigma_M[m] = rand(InverseGamma(model.alpha0 + model.Q/2, model.beta0 + sum((Tau_QM[:,m] .- model.taud .- model.tauc*Tau_QM[:,m-1]).^2)/2))
    #         # end
    #     end

    #     @views @threads for n in 1:model.N
    #         W = Diagonal(W_NM[n,:]) 
    #         V = inv(Tau_QM * W * Tau_QM' + (1/sigma2_beta)*Matrix{Int}(I, model.Q, model.Q))
    #         V = .5(V + V')
    #         k = (D_NM[n,:] .- 1)/2 .- (model.Dmax - 1)/4
    #         mvec = Float64.(V*(Tau_QM * k))
    #         Beta_NQ[n,:] = rand(MvNormal(mvec,V))

    #         #update variance
    #         #sigma_N[n] = rand(InverseGamma(model.alpha0 + model.Q/2, model.beta0 + sum(Beta_NQ[n,:].^2)/2))
    #     end

    #     # Update tied variances once per iteration
    #     sigma2_tau = rand(InverseGamma(
    #         model.alpha0 + (model.Q*model.M)/2,
    #         model.beta0 + sum((Tau_QM[:,1] .- model.taud .- model.tauc*fill(model.start_tau, model.Q)).^2)/2 +
    #                     sum([sum((Tau_QM[:,m] .- model.taud .- model.tauc*Tau_QM[:,m-1]).^2)/2 for m in 2:model.M])
    #     ))

    #     sigma2_beta = rand(InverseGamma(
    #         model.alpha0 + (model.Q*model.N)/2,
    #         model.beta0 + sum(Beta_NQ.^2)/2
    #     ))
    # end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps" => eps, 
    "D_NM"=>D_NM, "Tau_QM"=>Tau_QM, "Beta_NQ"=>Beta_NQ,
     "sigma2_beta"=>sigma2_beta, "sigma2_tau"=>sigma2_tau,
    "Y0_N"=>Y0_N,)
    #"Y_NMKplus2"=>Y_NMKplus2)
    return data, state
end

#forecasting code 

function predict(model::covid4, state, info, n, m, Ylast)
    eps = state["eps"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    D_NM = state["D_NM"]
    return rand(OrderStatistic(Poisson(Ylast + eps + sum(U_NK[n,:] .* V_KM[:,m])), D_NM[n,m],div(D_NM[n,m],2)+1))
end

function predict_x(model::covid4, state, info, n, mstart, Ystart, x)
    result = zeros(x)
    Ylast = Ystart
    m = mstart + 1
    for i in 1:length(result)
        result[i] = predict(model, state, info, n, m, Ylast)
        m += 1
        Ylast = result[i]
    end
    return result
end


function forecast(model::covid4, state, data, info, Ti)
    lastgamma = state["V_KM"][:,end]
    forecastGamma_KTi = zeros(model.K, Ti)
    for i in 1:Ti
        forecastGamma_KTi[:,i] = rand.(Gamma.(model.c*lastgamma .+ model.d, 1/model.c))
        lastgamma = copy(forecastGamma_KTi[:,i])
    end 
    Ylast = data["Y_NM"][:,end]

    mu_NTi = state["U_NK"] * forecastGamma_KTi

    Y_NTi = zeros(model.N,Ti)
    for i in 1:Ti
        mu = Ylast .+ state["eps"] .+ mu_NTi[:,i]
        Y_NTi[:,i] = rand.(OrderStatistic.(Poisson.(mu), model.D, model.j))
        Ylast = copy(Y_NTi[:,i])
    end
    return Y_NTi
end