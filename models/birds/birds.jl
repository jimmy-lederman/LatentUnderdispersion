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
    Dstar::Int64
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
    logprobs = [logpmfMaxPoisson(Y,mu,d) + logbinomial(model.Dstar-1, d-1) + (d-1)*log(p) + (model.Dstar - d)*log(1-p) for d in 1:model.Dstar]
    D = argmax(rand(Gumbel(0,1), model.Dstar) .+ logprobs)
    @assert D >= 1 && D <= model.Dstar
    return D
end

function sample_prior(model::birds,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    X_NP = info["X_NP"]
    P = size(X_NP)[2]
    Beta_PM = rand(Normal(0,1), P, model.M) #???

    
    D_NM = rand.(Binomial.(model.Dstar - 1, logistic.(X_NP*Beta_PM))) .+ 1

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_PM"=>Beta_PM, "D_NM"=>D_NM)
    return state
end

function forward_sample(model::birds; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    D_NM = state["D_NM"]

    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), D_NM, D_NM))
    #Y_NM = zeros(model.N, model.M)



    #Z_NMK = zeros(Int, model.N, model.M, model.K)
    # for n in 1:model.N
    #     for m in 1:model.M
    #         for k in 1:model.K
    #             Z_NMK[n,m,k] = rand(Poisson(state["U_NK"][n,k]*state["V_KM"][k,m]))
    #         end
    #     end
    # end
    data = Dict("Y_NM" => Y_NM)
    #state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"])
    return data, state 
end

function backward_sample(model::birds, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_PM = copy(state["Beta_PM"])
    D_NM = copy(state["D_NM"])
    
    X_NP = info["X_NP"]
    P = size(X_NP)[2]
    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    Mu_NM = U_NK * V_KM
   
    #Polya-gamma augmentation to update Beta
    F_NM = X_NP * Beta_PM
    W_NM = zeros(Float64, model.N, model.M)
    
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        pg = PolyaGammaHybridSampler(model.Dstar - 1, F_NM[n,m])
        W_NM[n,m] = rand(pg)
    end
    
    @views @threads for m in 1:model.M
        W = Diagonal(W_NM[:,m]) 
        
        V = inv(X_NP' * W * X_NP + Matrix{Int}(I, P, P))
        
        V = .5(V + V')
        k = D_NM[:,m] .- (model.Dstar - 1)/2 .- 1
       
        mvec = Float64.(V*(X_NP' * k))
        
        Beta_PM[:,m] = rand(MvNormal(mvec,V))
    end
    
    p_NM = logistic.(X_NP * Beta_PM)
    #D_NM = zeros(Int, model.N, model.M)

    #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
       
    #for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        
        D_NM[n,m] = sample_D(model, Y_NM[n,m], Mu_NM[n,m], p_NM[n,m])
        

        # if !isnothing(mask)
        #     if mask[n,m] == 1
        #         Y_NM[n,m] = rand(OrderStatistic(Poisson(Mu_NM[n,m]),  D_NM[n,m],  D_NM[n,m]))
        #     end
        # end
        
        if Y_NM[n, m] > 0
            Z_NM[n, m] = sampleSumGivenOrderStatistic(Y_NM[n, m], D_NM[n,m], D_NM[n,m], Poisson(Mu_NM[n, m]))
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z_NM[n, m], P_K / sum(P_K)))
        end
    end

    C_NK = D_NM * V_KM'
    Z_NK = dropdims(sum(Z_NMK, dims=2),dims=2)
    U_NK = rand.(Gamma.(model.a .+ Z_NK, 1 ./(model.b .+ C_NK)))

    C_KM = U_NK' *D_NM
    Z_KM = dropdims(sum(Z_NMK, dims=1),dims=1)'
    V_KM = rand.(Gamma.(model.c .+ Z_KM, 1 ./(model.d .+ C_KM)))


    # @views for n in 1:model.N
    #     @views for k in 1:model.K
    #         post_shape = model.a + sum(Z_NMK[n, :, k])
    #         post_rate = model.b + model.D*C_NK
    #         U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end

    # @views for m in 1:model.M
    #     @views for k in 1:model.K
    #         post_shape = model.c + sum(Z_NMK[:, m, k])
    #         post_rate = model.d + model.D*sum(U_NK[:, k])
    #         V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_PM"=>Beta_PM, "D_NM"=>D_NM)
    return data, state
end