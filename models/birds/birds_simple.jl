include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
include("../genes_polya/PolyaGammaHybridSamplers.jl/src/pghybrid.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct birds <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    P::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end


function evalulateLogLikelihood(model::birds, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    D = state["D_NM"][row,col]
    return logpmfMaxPoisson(Y,mu,D)
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function sample_D(model::birds, Y, mu, p)
    logprobs = [logpmfMaxPoisson(Y,mu,d) + logbinomial(model.Dmax-1, d-1) + (d-1)*log(p) + (model.Dmax - d)*log(1-p) for d in 1:model.Dmax]
    D = argmax(rand(Gumbel(0,1), model.Dmax) .+ logprobs)
    @assert D >= 1 && D <= model.Dmax
    return D
end

function sample_prior(model::birds,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    X_NP = info["X_NP"]
    P = size(X_NP)[2]
    Tau_NP = rand(Normal(0,1), model.N, model.P)
    Beta_PM = rand(Normal(0,1), model.P, model.M) 
    D_NM = rand.(Binomial.(model.Dmax - 1, logistic.(Tau_NP * Beta_PM))) .+ 1

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_PM"=>Beta_PM, "Tau_NP"=>Tau_NP, "D_NM"=>D_NM)
    return state
end

function forward_sample(model::birds; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    D_NM = state["D_NM"]

    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), D_NM, D_NM))
    data = Dict("Y_NM" => Y_NM)

    return data, state 
end

function backward_sample(model::birds, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_PM = copy(state["Beta_PM"])
    Tau_NP = copy(state["Tau_NP"])

    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    Mu_NM = U_NK * V_KM
    D_NM = copy(state["D_NM"])
   
    #Polya-gamma augmentation to update Beta
    F_NM = Tau_NP * Beta_PM
    W_NM = zeros(Float64, model.N, model.M)
    
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        pg = PolyaGammaHybridSampler(model.Dmax - 1, F_NM[n,m])
        W_NM[n,m] = rand(pg)
    end
    
    @views @threads for m in 1:model.M
        W = Diagonal(W_NM[:,m]) 
        V = inv(Tau_NP' * W * Tau_NP + Matrix{Int}(I, model.P, model.P))
        V = .5(V + V')
        k = D_NM[:,m] .- (model.Dmax - 1)/2 .- 1
        mvec = Float64.(V*(Tau_NP' * k))
        Beta_PM[:,m] = rand(MvNormal(mvec,V))
    end

    @views @threads for n in 1:model.N
        W = Diagonal(W_NM[n,:]) 
        V = inv(Beta_PM * W * Beta_PM' + Matrix{Int}(I, model.P, model.P))
        V = .5(V + V')
        k = D_NM[n,:] .- (model.Dmax - 1)/2 .- 1
        mvec = Float64.(V*(Beta_PM * k))
        Tau_NP[n,:] = rand(MvNormal(mvec,V))
    end
    
    p_NM = logistic.(Tau_NP * Beta_PM)
    #D_NM = zeros(Int, model.N, model.M)

    #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
       
    #for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        
        D_NM[n,m] = sample_D(model, Y_NM[n,m], Mu_NM[n,m], p_NM[n,m])
        

        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]),  D_NM[n,m],  D_NM[n,m]))
            end
        end
        
        if Y_NM[n, m] > 0
            Z_NM[n, m] = sampleSumGivenOrderStatistic(Y_NM[n, m], D_NM[n,m], D_NM[n,m], Poisson(Mu_NM[n, m]))
            if Z_NM[n,m] > 0
                P_K = U_NK[n, :] .* V_KM[:, m]
                Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K / sum(P_K)))
            end
        end
    end

    C_NK = D_NM * V_KM'
    Z_NK = dropdims(sum(Z_NMK, dims=2),dims=2)
    U_NK = rand.(Gamma.(model.a .+ Z_NK, 1 ./(model.b .+ C_NK)))

    C_KM = U_NK' *D_NM
    Z_KM = dropdims(sum(Z_NMK, dims=1),dims=1)'
    V_KM = rand.(Gamma.(model.c .+ Z_KM, 1 ./(model.d .+ C_KM)))

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_PM"=>Beta_PM, "Tau_NP"=>Tau_NP, "D_NM"=>D_NM)
    return data, state
end