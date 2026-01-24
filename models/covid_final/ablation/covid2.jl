include("../../../helper/MatrixMF.jl")
include("../../../helper/OrderStatsSampling.jl")
include("../../../helper/PoissonOrderPMF.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
using SpecialFunctions
using Base.Threads


struct covid1 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end


function evalulateLogLikelihood(model::covid1, state, data, info, row, col)
    @assert !isnothing(info)
    if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]
    D = state["D"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast + mu
    if D == 1
        return logpdf(Poisson(rate), Y)
    else
        return logpmfOrderStatPoisson(Y,rate,D,div(D,2)+1)
    end
end

function sample_likelihood(model::covid1, mu,D,n=1)
    if D == 1
        return rand(Poisson(mu),n)
    else
        return rand(OrderStatistic(Poisson(mu),D, div(D,2)+1),n)
    end
end

function sample_prior(model::covid1,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)

    D = 2*rand(Binomial(Int((model.Dmax - 1)/2), .5)) .+ 1

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM,
    "D"=>D,
     "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid1; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Y0_N = state["Y0_N"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    D = state["D"]
    p = state["p"]

    Y_NM = zeros(Int, model.N, model.M)

    for m in 1:model.M
        for n in 1:model.N
            if m == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y0_N[n] + sum(U_NK[n,:] .* V_KM[:,m])), D, div(D,2)+1))
            else
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + sum(U_NK[n,:] .* V_KM[:,m])),D, div(D,2)+1))
            end
        end 
    end

    data = Dict("Y_NM" => Y_NM)

    return data, state 
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


function update_Dall(model::covid1, Y0_N, Y_NM, mu_NM, p)
    logprobs = [logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) + (d-1)*log(p)/2 + (model.Dmax - d)*log(1-p)/2 for d in 1:2:model.Dmax]
    @views @threads for n in 1:model.N
        @views for m in 1:model.M
            if m == 1
                mu = Y0_N[n] + mu_NM[n,m]
            else
                mu = Y_NM[n,m] + mu_NM[n,m]
            end
            Y = Y_NM[n,m]
            logprobs += [logpmfOrderStatPoisson(Y,mu,d,div(d,2)+1,compute=false) for d in 1:2:model.Dmax]
        end
    end
    D = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
    @assert D >= 1 && D <= model.Dmax
    return D
end

function backward_sample(model::covid1, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    D = copy(state["D"])
    Y0_N = copy(state["Y0_N"])

    Y_NMKplus2 = zeros(model.N, model.M, model.K + 1)
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(model.N * model.M)
        m = div(idx - 1, model.N) + 1
        n = mod(idx - 1, model.N) + 1
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        probvec = U_NK[n,:] .* V_KM[:,m]
        mu1 = sum(probvec) 
        mu = Ylast + mu1

        if !isnothing(mask)
            if mask[n,m] == 1   
                Y_NM[n,m] = rand(OrderStatistic(Poisson(mu), D, div(D, 2) + 1))
            end
        end

        Z = sampleSumGivenOrderStatistic(Y_NM[n, m], D, div(D,2)+1, Poisson(mu))
        if Z > 0 
            probvec = vcat(probvec, Ylast)
            Y_NMKplus2[n, m, :] = rand(Multinomial(Z, probvec / sum(probvec)))
        end
    end

    #update county factors
    # Y_NK = dropdims(sum(Y_NMKplus2,dims=2),dims=2)[:,1:model.K]
    # C2_NK = D * sum(V_KM, dim = 2)
    # U_NK = rand.(Gamma.(model.a .+ Y_NK, 1 ./(model.b .+ C2_NK)))

    @views for n in 1:model.N
        @views for k in 1:model.K
            post_shape = model.a + sum(Y_NMKplus2[n, :, k])
            post_rate = model.b + D*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Y_NMKplus2[:, m, k])
            post_rate = model.d + D*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    
    #Polya-gamma augmentation to update D
    if isnothing(skipupdate) || !("D" in skipupdate)
        D = update_Dall(model, Y0_N, Y_NM, U_NK * V_KM, .5)
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "D" => D, "Y0_N"=>Y0_N,)
    return data, state
end