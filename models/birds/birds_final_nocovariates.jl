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
    Q::Int64
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
    logprobs = [logpmfMaxPoisson(Y,mu,d,compute=false) + logbinomial(model.Dmax-1, d-1) + (d-1)*log(p) + (model.Dmax - d)*log(1-p) for d in 1:model.Dmax]
    # if sum(isinf.(logprobs)) > 0
       #println(logprobs)
    #     @assert 1 == 2
    # end
    D = argmax(rand(Gumbel(0,1), model.Dmax) .+ logprobs)
    @assert D >= 1 && D <= model.Dmax
    return D
end

function sample_prior(model::birds,info=nothing, constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    # Tau_NQ = rand(Normal(0,sqrt(model.sigma2)), model.N, model.Q)
    # Tau_N = rand(Normal(1,sqrt(model.sigma2)), model.N)
    # Beta_QM = rand(Normal(0,sqrt(model.sigma2)), model.Q, model.M) 
    # Beta_M = rand(Normal(model.constant, sqrt(model.sigma2)), model.M)
    Tau_NQ = rand(Normal(0,1), model.N, model.Q)
    # Tau_N = rand(Normal(1,sqrt(model.sigma2)), model.N)
    Beta_QM = rand(Normal(0,1), model.Q, model.M) 
    # Beta_M = rand(Normal(model.constant, sqrt(model.sigma2)), model.M)
    # Tau_NQp1 = hcat(Tau_NQ, Tau_N)
    # Beta_Pp1M = hcat(Beta_QM', Beta_M)'
    D_NM = rand.(Binomial.(model.Dmax - 1, logistic.(Tau_NQ * Beta_QM))) .+ 1

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_QM"=>Beta_QM, "Tau_NQ"=>Tau_NQ, #"Beta_M"=>Beta_M, "Tau_N"=>Tau_N, 
    "D_NM"=>D_NM)
    return state
end

function forward_sample(model::birds; state=nothing, info=nothing, constantinit=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    D_NM = state["D_NM"]

    Y_NM = rand.(OrderStatistic.(Poisson.(Mu_NM), D_NM, D_NM))
    data = Dict("Y_NM" => Y_NM)

    return data, state 
end

function backward_sample(model::birds, data, state, mask=nothing;skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_QM = copy(state["Beta_QM"])
    # Beta_M = copy(state["Beta_M"])
    Tau_NQ = copy(state["Tau_NQ"])
    # Tau_N = copy(state["Tau_N"])

    Z_NM = zeros(Int, model.N, model.M)
    Z_NMK = zeros(Int, model.N, model.M, model.K)
    Mu_NM = U_NK * V_KM
    D_NM = copy(state["D_NM"])

    #Loop over the non-zeros in Y_DV and allocate
    @views @threads for idx in 1:(model.N * model.M)
       
    #for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1      

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

    C_KM = U_NK' * D_NM
    Z_KM = dropdims(sum(Z_NMK, dims=1),dims=1)'
    V_KM = rand.(Gamma.(model.c .+ Z_KM, 1 ./(model.d .+ C_KM)))

    if isnothing(skipupdate) || !("D_NM" in skipupdate)
        #Polya-gamma augmentation to update Beta
        F_NM = Tau_NQ * Beta_QM
        p_NM = logistic.(F_NM)
        W_NM = zeros(Float64, model.N, model.M)
        Mu_NM = U_NK * V_KM
        @views @threads for idx in 1:(model.N * model.M)
            n = div(idx - 1, model.M) + 1
            m = mod(idx - 1, model.M) + 1      

            D_NM[n,m] = sample_D(model, Y_NM[n,m], Mu_NM[n,m], p_NM[n,m])
            pg = PolyaGammaHybridSampler(model.Dmax - 1, F_NM[n,m])
            W_NM[n,m] = rand(pg)
        end
        # Tau_NQp1 = hcat(TauNP, Tau_N)
        # Beta_Pp1M = vcat(Beta_M, Beta_QM)
        ident =  Matrix{Int}(I, model.Q, model.Q)
        @views @threads for m in 1:model.M
            W = Diagonal(W_NM[:,m]) 
            V = inv(Tau_NQ' * W * Tau_NQ + ident)
            V = .5(V + V')
            k = D_NM[:,m] .- (model.Dmax - 1)/2 .- 1
            mvec = Float64.(V*(Tau_NQ' * k))
            Beta_QM[:,m] = rand(MvNormal(mvec,V))
        end
        # offset = zeros(model.Q+1)
        # offset[end] = 1/model.sigma2
        @views @threads for n in 1:model.N
            W = Diagonal(W_NM[n,:]) 
            V = inv(Beta_QM * W * Beta_QM' + ident)
            V = .5(V + V')
            k = D_NM[n,:] .- (model.Dmax - 1)/2 .- 1
            mvec = Float64.(V*(Beta_QM * k))
            Tau_NQ[n,:] = rand(MvNormal(mvec,V))
        end
    
        
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Beta_QM"=>Beta_QM, "Tau_NQ"=>Tau_NQ, "D_NM"=>D_NM)
    return data, state
end