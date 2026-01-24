include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
using LinearAlgebra
using Base.Threads
include("../genes_polya/PolyaGammaHybridSamplers.jl/src/pghybrid.jl")

using LogExpFunctions

struct genes <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    Q::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha0::Float64
    beta0::Float64
    alpha::Float64
    beta::Float64
    constant::Float64
    #model.sigma2::Float64
end

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        return 1
    else
        probs = [R/(R+i-1) for i in 2:Y]
    end
    return 1 + sum(rand.(Bernoulli.(probs)))
end

function sampleCRTlecam(Y,R,tol=.4)
    Y_max = R * (1/tol - 1)
    if Y <= Y_max || Y <= 100
        return sampleCRT(Y, R)
    else
        out = sampleCRT(Y_max, R)
        mu = R * (polygamma(0, R + Y) - polygamma(0, R + Y_max))
        return out + rand(Poisson(mu))
    end
end

function evalulateLogLikelihood(model::genes, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    r = sum(state["U_NK"][row,:] .* state["V_KM"][:,col])
    p = state["p_N"][row]
    D = state["D_NM"][row,col]

    if D == 1
        return logpdf(NegativeBinomial(r,p), Y)
    else
        return logpmfOrderStatNegBin(Y, r, p, D, div(D,2)+1)
    end
end

function sample_prior(model::genes,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    
    sigma_N = rand(InverseGamma(model.alpha0, model.beta0), model.N)
    sigma_M = rand(InverseGamma(model.alpha0, model.beta0), model.M)

    Beta_NQ = hcat(rand.(Normal.(0,sqrt.(sigma_N)),model.Q)...)'
    Tau_QM = hcat(rand.(Normal.(0,sqrt.(sigma_M)),model.Q)...)

    #Beta_N = rand.(Normal.(model.constant, sqrt.(sigma_N)))
    #Beta_N = rand(Normal(model.constant, 1), model.N)
    #Tau_M = rand(Normal(1, sqrt(model.sigma2)), model.MatrixMF)
    #Tau_M = fill(1, model.M)

    Beta_NQp1 = hcat(Beta_NQ, Beta_N)
    Tau_Qp1M = hcat(Tau_QM', Tau_M)'

    p_N = rand(Beta(model.alpha, model.beta), model.N)

    p_NM = logistic.(Beta_NQp1 * Tau_Qp1M)
    D_NM = 2*rand.(Binomial.(model.Dmax - 1, p_NM)) .+ 1
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D_NM" => D_NM, "Beta_NQp1"=>Beta_NQp1, "Tau_Qp1M"=>Tau_Qp1M,
     "p_N"=>p_N,"sigma_N"=>sigma_N,"sigma_M"=>sigma_M)
end

function sample_likelihood(model::genes, mu,p,D)
    if D == 1
        return rand(NegativeBinomial(mu,p))
    else
        return rand(OrderStatistic(NegativeBinomial(mu,p), D, div(D,2)+1))
    end
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function update_Dmax(model::genes, Y, r, p1, p2)
    logprobs = [logpmfMaxNegBin(Y,r,p1,d,compute=false) + logbinomial(model.Dmax-1, d-1) + (d-1)*log(p2) + (model.Dmax - d)*log(1-p2) for d in 1:model.Dmax]
    D = argmax(rand(Gumbel(0,1), model.Dmax) .+ logprobs)
    @assert D >= 1 && D <= model.Dmax
    return D
end

function update_Dmedian(model::genes, Y, r, p1, p2)
    logprobs = [logpmfOrderStatNegBin(Y,r,p1,d,div(d,2)+1,compute=false) + logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(p2)/2 + (model.Dmax - d)*log(1-p2)/2 for d in 1:2:model.Dmax]
    D = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
    @assert D >= 1 && D <= model.Dmax
    return D
end

function forward_sample(model::genes; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    # Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = zeros(Int, model.N, model.M)
    #Z_NMK = zeros(Int, model.N, model.M, model.K)
    p_N = copy(state["p_N"])
    D_NM = copy(state["D_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])

    for n in 1:model.N
        for m in 1:model.M
            Y_NM[n,m] = sample_likelihood(model,sum(U_NK[n,:].*V_KM[:,m]),p_N[n],D_NM[n,m])
        end
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::genes, data, state, mask=nothing;skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    #Ysparse = copy(data["Ysparse"])
    p_N = copy(state["p_N"])
    D_NM = copy(state["D_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_NQp1 = copy(state["Beta_NQp1"])
    Tau_Qp1M = copy(state["Tau_Qp1M"])
    sigma_N = copy(state["sigma_N"])
    sigma_M = copy(state["sigma_M"])



    Z1_NM = zeros(Int, model.N, model.M)
    Z2_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
   
    Mu_NM = U_NK * V_KM
    
     #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        mu = Mu_NM[n,m]
        p = p_N[n]
        D = D_NM[n,m]
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = sample_likelihood(model, mu, p, D) 
            end
        end
        
        Z2_NM[n,m] = sampleSumGivenOrderStatistic(Y_NM[n,m], D, div(D,2)+1, NegativeBinomial(mu,p))
        if Z2_NM[n,m] > 0
            Z1_NM[n,m] = sampleCRTlecam(Z2_NM[n,m], D*mu)
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z1_NM[n, m], P_K / sum(P_K)))
        end
    end

    #update p_N with Beta conjugacy
    post_alpha = model.alpha .+ sum(D_NM .* Mu_NM, dims=2) 
    post_beta = model.beta .+ sum(Z2_NM, dims=2)
    p_N = rand.(Beta.(post_alpha,post_beta))

    Z_NK = dropdims(sum(Z_NMK,dims=2), dims=2)
    C_NK = log.(1 ./p_N) .*  D_NM * V_KM'
    U_NK = rand.(Gamma.(model.a .+ Z_NK, 1 ./(model.b .+ C_NK)))
    
    Z_KM = dropdims(sum(Z_NMK,dims=1), dims=1)'
    C_KM = (log.(1 ./ p_N) .* U_NK)' * D_NM
    V_KM = rand.(Gamma.(model.c .+ Z_KM, 1 ./(model.d .+ C_KM)))
    
    
    #Update D
    
    if isnothing(skipupdate) || !("D_NM" in skipupdate)
        
        #Polya-gamma augmentation to update Beta
        F_NM = Beta_NQp1 * Tau_Qp1M
        #F_NM2 = Beta_NQp1[:,1:end-1] * Tau_Qp1M[1:end-1,:]
        p_NM = logistic.(F_NM)
        W_NM = zeros(Float64, model.N, model.M)
        #W_NM2 = zeros(Float64, model.N, model.M)
        Mu_NM = U_NK * V_KM
        @views @threads for idx in 1:(model.N * model.M)
            n = div(idx - 1, model.M) + 1
            m = mod(idx - 1, model.M) + 1 

            D_NM[n,m] = update_Dmedian(model, Y_NM[n,m], Mu_NM[n,m], p_N[n], p_NM[n,m])

            pg = PolyaGammaHybridSampler((model.Dmax - 1)/2, F_NM[n,m])
            #pg2 = PolyaGammaHybridSampler(model.Dmax - 1, F_NM2[n,m])
            W_NM[n,m] = rand(pg)
            #W_NM2[n,m] = rand(pg)
        end
        
        # offset = zeros(model.Q+1)
        # offset[end] = 1/model.sigma2

        Beta_NQ = Beta_NQp1[:,1:(end-1)]
        @views for m in 1:model.M
            W = Diagonal(W_NM[:,m]) 
            ident =  (1/sigma_M[m])*Matrix{Int}(I, model.Q, model.Q)
            #ident[end,end] = 1/model.sigma2
            V = inv(Beta_NQ' * W * Beta_NQ + ident)
            V = .5(V + V')
            k = D_NM[:,m] .- (model.Dmax - 1)/2 .- 1
            offset = fill(sum(Beta_NQp1[:,end])/sigma_N[m], model.Q)
            #offset[end] = 
            mvec = Float64.(V*(Beta_NQ' * k + offset))
            Tau_Qp1M[:,m] = vcat(rand(MvNormal(mvec,V)),1)
            
            #update variance
            sigma_M[m] = rand(InverseGamma(model.alpha0 + model.Q/2, model.beta0 + sum(Tau_Qp1M[1:(end-1),m].^2)/2))
        end
        # Tau_Qp1M[end,:] = fill(1,model.M)
        # println(Tau_Qp1M)
        @views for n in 1:model.N
            W = Diagonal(W_NM[n,:]) 
            ident = (1/sigma_N[n])*Matrix{Int}(I, model.Q+1, model.Q+1)
            # ident[end,end] = 1 
            V = inv(Tau_Qp1M * W * Tau_Qp1M' + ident)
            V = .5(V + V')
            k = D_NM[n,:] .- (model.Dmax - 1)/2 .- 1
            offset = zeros(model.Q+1)
            offset[end] = model.constant/sigma_N[n]
            mvec = Float64.(V*(Tau_Qp1M * k + offset))
            Beta_NQp1[n,:] = rand(MvNormal(mvec,V))
            #update variance

            sigma_N[n] = rand(InverseGamma(model.alpha0 + (model.Q+1)/2, model.beta0 + sum((offset .- Beta_NQp1[n,1:end]).^2)/2))
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_NQp1"=>Beta_NQp1, "Tau_Qp1M"=>Tau_Qp1M,
    "sigma_N"=>sigma_N,"sigma_M"=>sigma_M,"D_NM"=>D_NM,"p_N"=>p_N)
    return data, state
end