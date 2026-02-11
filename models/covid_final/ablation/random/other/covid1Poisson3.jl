include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct covid1Poisson3 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::covid1Poisson3, state, data, info, row, col)
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

function sample_prior(model::covid1Poisson3,info=nothing,constantinit=nothing)
    U_NK = rand(Dirichlet(ones(model.N)), model.K)
    phi_KM = rand(Dirichlet(fill(model.c, model.M)), model.K)'
    a_K = rand(Gamma(model.M*model.c, 1/model.d), model.K)

    V_KM = zeros(model.K, model.M)
    for k in 1:model.K
        V_KM[k,:] = a_K[k]*phi_KM[k,:]
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "a_K"=>a_K, "phi_KM"=>phi_KM, "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid1Poisson3; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
    end
    Mu_NM = state["U_NK"] * Diagonal(state["a_K"])  * state["phi_KM"]
    Y_NM = rand.(Poisson.(Mu_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid1Poisson3, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    a_K = copy(state["a_K"])
    phi_KM = copy(state["phi_KM"])
    Y0_N = copy(state["Y0_N"])
    Y_NMK = zeros(model.N, model.M, model.K + 1)
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        P_K = U_NK[n,:] .* a_K .* phi_KM[:,m]
        P_K = vcat(P_K, Ylast)
        mu = sum(P_K)
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(Poisson(mu))
            end
        end
        if Y_NM[n, m] > 0
            #P_K = U_NK[n, :] .* V_KM[:, m]
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            Y_NMK[n, m, :] = y_k
        end
    end
    @assert sum(Y_NMK) == sum(Y_NM)
    
    Y_NK = dropdims(sum(Y_NMK, dims = 2), dims = 2)
    @views for k in 1:model.K
        U_NK[:, k] = rand(Dirichlet(ones(model.N) .+ Y_NK[:,k]))
    end

    phi_KM = zeros(model.K, model.M)
    Y_MK = dropdims(sum(Y_NMK, dims = 1), dims = 1)
    @views for k in 1:model.K
        phi_KM[k, :] = rand(Dirichlet(fill(model.c, model.M) .+ Y_MK[:,k]))
    end

    a_K = zeros(model.K)
    Y_K = dropdims(sum(Y_MK, dims = 1), dims = 1)
    #U_K = dropdims(sum(U_NK, dims = 1), dims = 1)
    @views for k in 1:model.K 
        a_K[k] = rand(Gamma(model.c*model.M + Y_K[k], 1/(model.d + 1)))
    end

    V_KM = zeros(model.K, model.M)
    for k in 1:model.K
        V_KM[k,:] = a_K[k]*phi_KM[k,:]
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "a_K" => a_K, "phi_KM" => phi_KM, "Y0_N"=>info["Y0_N"])
    return data, state
end
