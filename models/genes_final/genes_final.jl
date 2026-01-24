include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
using LinearAlgebra
using Base.Threads
include("../genes_polya/PolyaGammaHybridSamplers.jl/src/pghybrid.jl")

using LogExpFunctions

struct genes_base <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
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

function evalulateLogLikelihood(model::genes_base, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    r = sum(state["U_NK"][row,:] .* state["V_KM"][:,col])
    p = state["p_N"][row]
    return logpdf(NegativeBinomial(r,p), Y)
end

function sample_prior(model::genes_base,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)

    p_N = rand(Beta(model.alpha, model.beta), model.N)

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p_N"=>p_N)
end

function sample_likelihood(model::genes_base, mu,p)
    return rand(NegativeBinomial(mu,p))
end

function forward_sample(model::genes_base; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Y_NM = zeros(Int, model.N, model.M)
    p_N = copy(state["p_N"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])

    for n in 1:model.N
        for m in 1:model.M
            Y_NM[n,m] = sample_likelihood(model,sum(U_NK[n,:].*V_KM[:,m]),p_N[n])
        end
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::genes_base, data, state, mask=nothing;skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    p_N = copy(state["p_N"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])


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
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = sample_likelihood(model, mu, p) 
            end
        end
        if Y_NM[n,m] > 0
            Z1_NM[n,m] = sampleCRTlecam(Y_NM[n,m], mu)
            P_K = U_NK[n, :] .* V_KM[:, m]
            Z_NMK[n, m, :] = rand(Multinomial(Z1_NM[n, m], P_K / sum(P_K)))
        end
    end

    #update p_N with Beta conjugacy
    post_alpha = model.alpha .+ sum(Mu_NM, dims=2) 
    post_beta = model.beta .+ sum(Y_NM, dims=2)
    p_N = rand.(Beta.(post_alpha,post_beta))

    Z_NK = dropdims(sum(Z_NMK,dims=2), dims=2)
    C_NK = log.(1 ./p_N) * sum(V_KM,dims=2)'
    U_NK = rand.(Gamma.(model.a .+ Z_NK, 1 ./(model.b .+ C_NK)))
    
    Z_KM = dropdims(sum(Z_NMK,dims=1), dims=1)'
    C_K = dropdims(sum(log.(1 ./ p_N) .* U_NK,dims=1),dims=1)
    C_KM = repeat(C_K, 1, model.M)
    V_KM = rand.(Gamma.(model.c .+ Z_KM, 1 ./(model.d .+ C_KM)))

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM,"p_N"=>p_N)
    return data, state
end