include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct STARMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    tau_phi2::Float64
    tau_theta2::Float64
    g::Function
    g_inv::Function
end

function STARlogpmf(model::STARMF, x,mu,sigma2)
    if x == 0
        return log(cdf(Normal(0,1), (0.5 - mu)/sqrt(sigma2)))
    else 
        return log(cdf(Normal(0,1), (model.g(x + .5) - mu)/sqrt(sigma2)) - cdf(Normal(0,1), (model.g(x - .5) - mu)/sqrt(sigma2)))
    end
end

function evalulateLogLikelihood(model::STARMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    sigma2 = state["sigma2"]
    return STARlogpmf(model,Y,mu,sigma2)
end

function sample_prior(model::STARMF,info=nothing)
    U_NK = rand(Normal(0, sqrt(model.tau_phi2)), model.N, model.K)
    V_KM = rand(Normal(0, sqrt(model.tau_theta2)), model.K, model.M)
    sigma2 = rand(InverseGamma(model.a, model.b))
    sigma_NM = fill(sqrt(sigma2), model.N, model.M)
    Z_NM = rand.(Normal.(U_NK*V_KM, sigma_NM))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Z_NM"=>Z_NM, "sigma2"=>sigma2)
    return state
end

function forward_sample(model::STARMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Y_NM = zeros(model.N, model.M)
    Z_NM = state["Z_NM"] 
    for n in 1:model.N
        for m in 1:model.M
            if Z_NM[n,m] < .5
                Y_NM[n,m] = 0
            else
                Y_NM[n,m] =  Int(round(model.g_inv(Z_NM[n,m])))
            end
        end
    end


    
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

function backward_sample(model::STARMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NM = copy(state["Z_NM"])
    # Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    Mu_NM = U_NK * V_KM

    sigma2 = copy(state["sigma2"])
    #Z_NM = zeros(model.N,model.M)

    @views @threads for n in 1:model.N
        @views for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    z = rand.(Normal.(Mu_NM[n,m], sqrt(sigma2)))
                    if z < .5
                        Y_NM[n,m] = 0
                    else
                        Y_NM[n,m] =  Int(round(model.g_inv(z)))
                    end
                end
            end
            if Y_NM[n,m] == 0
                Z_NM[n,m] = rand(Truncated(Normal(Mu_NM[n,m], sqrt(sigma2)), -Inf, model.g(.5)))
            else
                Z_NM[n,m] = rand(Truncated(Normal(Mu_NM[n,m], sqrt(sigma2)), model.g(Y_NM[n,m] - .5), model.g(Y_NM[n,m] + .5)))
            end
        end
    end

        # Update Phi
    @views for n in 1:model.N
        for k in 1:model.K
            R = Z_NM[n, :] .- (vec(U_NK[n, :]' * V_KM)) .+ (vec(U_NK[n, k] * V_KM[k, :]))
            sigma2_phi = 1 / (sum(V_KM[k, :].^2)/sigma2 + 1/model.tau_phi2)
            mu_phi = sigma2_phi * sum(V_KM[k, :] .* R)/sigma2
            U_NK[n, k] = rand(Normal(mu_phi, sqrt(sigma2_phi)))
        end
    end

    # Update Theta
    @views for k in 1:model.K
        for m in 1:model.M
            R = Z_NM[:, m] .- U_NK * V_KM[:, m] + U_NK[:, k] * V_KM[k, m]
            sigma2_theta = 1 / (sum(U_NK[:, k].^2)/sigma2 + 1/model.tau_theta2)
            mu_theta = sigma2_theta * sum(U_NK[:, k] .* R)/sigma2
            V_KM[k, m] = rand(Normal(mu_theta, sqrt(sigma2_theta)))
        end
    end

    # Update sigma^2
    # post_a = model.a + model.N/2
    # post_b = model.b + sum((Z_N .- mu).^2)/2
    # sigma2 = rand(InverseGamma(post_a, post_b))
    resid = Z_NM .- U_NK * V_KM
    sigma2 = rand(InverseGamma(model.a + model.N*model.M/2, (model.b + sum(resid.^2)/2)))

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2"=>sigma2, "Z_NM"=>Z_NM)
    return data, state
end
