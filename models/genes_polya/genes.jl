include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling_old.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
using LinearAlgebra
using Base.Threads
#using Suppressor
include("PolyaGammaHybridSamplers.jl/src/pghybrid.jl")

using LogExpFunctions

struct genes <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    Q::Int64
    a::Float64
    b::Float64
    c::Float64x2
    d::Float64
    D::Int64
    j::Int64
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
    p = logistic(sum(state["Beta_NQ"][row,:] .* state["Tau_QM"][:,col]))

    if model.D == 1
        return logpdf(NegativeBinomial(r,p), Y)
    elseif model.D == model.j
        return logpmfMaxNegBin(Y, r, p, model.D)
    else
        return logpmfOrderStatNegBin(Y, r, p, model.D, model.j)
    end
end

function sample_prior(model::genes,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)

    Beta_NQ = rand(Normal(0,1), model.N, model.Q)
    Tau_QM = rand(Normal(0,1), model.Q, model.M)
    p_NM = logistic.(Beta_NQ * Tau_QM)
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p_NM" => p_NM, "Beta_NQ"=>Beta_NQ, "Tau_QM"=>Tau_QM)
end

function sample_likelihood(model::genes, mu,p=nothing)
    if model.D == 1
        try
            return rand(NegativeBinomial(mu,p))
        catch ex 
            println(mu, " ", p)
            throw(ErrorException("could not sample D=1 lik"))
        end
    else
        return rand(OrderStatistic(NegativeBinomial(mu,p), model.D, model.j))
    end
end

function forward_sample(model::genes; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    # Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = zeros(Int, model.N, model.M)
    #Z_NMK = zeros(Int, model.N, model.M, model.K)
    p_NM = state["p_NM"]
    for n in 1:model.N
        for m in 1:model.M
            Y_NM[n,m] = sample_likelihood(model,sum(state["U_NK"][n,:].*state["V_KM"][:,m]),p_NM[n,m])
        end
    end
    data = Dict("Y_NM" => Y_NM)
    #state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"], "p_NM"=>p_NM, "Beta_NQ"=>Beta_NQ, "Tau_QM"=>Tau_QM)
    return data, state 
end

function backward_sample(model::genes, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    #Ysparse = copy(data["Ysparse"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_NQ = copy(state["Beta_NQ"])
    Tau_QM = copy(state["Tau_QM"])
    #p_NM = state["p_NM"]
    p_NM = logistic.(Beta_NQ * Tau_QM)
    @assert sum(p_NM .> 1) .== 0

    Z1_NM = zeros(Int, model.N, model.M)
    Z2_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
   
    Mu_NM = U_NK * V_KM

     #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        mu = Mu_NM[n,m]
        p = p_NM[n,m]
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = sample_likelihood(model, mu, p) 
            end
        end

        Z2_NM[n,m] = sampleSumGivenOrderStatistic(Y_NM[n,m], model.D, model.j, NegativeBinomial(mu,p))
        if Z2_NM[n,m] > 0
            Z1_NM[n,m] = sampleCRTlecam(Z2_NM[n,m], model.D*mu)
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z1_NM[n, m], P_K / sum(P_K)))
        end
        #end
    end
    
    #Polya-gamma augmentation to update p
    F_NM = Beta_NQ * Tau_QM
    W_NM = zeros(Float64, model.N, model.M)
    
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
    # @views @threads for n in 1:model.N
    #     @views for m in 1:model.M
            #sample a polya-gamma random variable
        pg = PolyaGammaHybridSampler(Z2_NM[n,m] + model.D*Mu_NM[n,m], F_NM[n,m])
        W_NM[n,m] = rand(pg)
    end

    #update Tau_QM
    
    @views for m in 1:model.M
        W = Diagonal(W_NM[:,m])
        V = inv(Beta_NQ' * W * Beta_NQ + I)
        V = .5(V + V')
        k = (.5*(model.D*Mu_NM[:,m] - Z2_NM[:,m]))
        
        mvec = V*(Beta_NQ' * k)
        
        Tau_QM[:,m] = rand(MvNormal(mvec,V))

    end

    #update Beta_NQ
    @views for n in 1:model.N
        W = Diagonal(W_NM[n,:])
        V = inv(Tau_QM * W * Tau_QM' + I)
        V = .5(V + V')
        k = .5*(model.D*Mu_NM[n,:] - Z2_NM[n,:])
        mvec = V*(Tau_QM * k)
        Beta_NQ[n,:] = rand(MvNormal(mvec,V))
    end

    p_NM = logistic.(Beta_NQ * Tau_QM)
    # println(minimum(p_NM))

    # @time @views @threads for idx in 1:(model.N*model.K)
    #     n = div(idx - 1, model.K) + 1
    #     k = mod(idx - 1, model.K) + 1 
    #     post_shape = model.a + sum(Z_NMK[n, :, k])
    #     post_rate = model.b + model.D*sum(log.(1 ./p_NM[n,:]) .* V_KM[k, :])
    #     U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
    # end
    Z_NK = dropdims(sum(Z_NMK,dims=2), dims=2)
    C_NK = model.D * log.(1 ./p_NM) * V_KM'
    @views  for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + Z_NK[n,k]#sum(Z_NMK[n, :, k])
            post_rate = model.b + C_NK[n,k]#model.D*sum(log.(1 ./p_NM[n,:]) .* V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end
    Z_MK = dropdims(sum(Z_NMK,dims=1), dims=1)
    C_MK = model.D * log.(1 ./ p_NM)' * U_NK
    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + Z_MK[m,k]#sum(Z_NMK[:, m, k])
            post_rate = model.d + C_MK[m,k]#model.D*sum(log.(1 ./ p_NM[:,m]) .* U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_NQ"=>Beta_NQ, "Tau_QM"=>Tau_QM)
    return data, state
end