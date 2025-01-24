include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
using SpecialFunctions
using Base.Threads

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        return 1
    else
        probs =[R/(R+i-1) for i in 2:Y]
        return 1 + sum(rand.(Bernoulli.(probs)))
    end
    
end


struct covidsimple <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    g::Float64 
    h::Float64
    scaleshape::Float64
    scalerate::Float64
    starta::Float64
    startb::Float64
    D::Int64
    j::Int64
end


function evalulateLogLikelihood(model::covidsimple, state, data, info, row, col)
    @assert !isnothing(info)
    if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]
    pop = info["pop_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+alpha*pop*mu+pop*eps
    if model.D == 1
        return logpdf(Poisson(rate), Y)
    else
        return logpmfOrderStatPoisson(Y,rate,model.D,model.j)
    end
end



function sample_prior(model::covidsimple,info=nothing,constantinit=nothing)
    pass = false
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    # V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.starta,1/model.startb))
                else
                    V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1]+model.d,1/model.c))
                end
            end
        end
    end
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, #"R_KTS" => R_KTS, 
    "eps"=>eps, "alpha" => alpha,
     "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"])
    return state
end

function forward_sample(model::covidsimple; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    eps = state["eps"]
    Y0_N = state["Y0_N"]
    pop_N = state["pop_N"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    #Y_NMKplus2 = zeros(Int, model.N, model.M, model.K + 2)
    Y_NM = zeros(Int, model.N, model.M)
    
    # for m in 1:model.M
    #     for n in 1:model.N
    #         for k in 1:(model.K+2)
    #             if k == model.K+1
    #                 if m == 1 
    #                     Y_NMKplus2[n,m,k] = rand(Poisson(Y0_N[n]))
    #                 else
    #                     Y_NMKplus2[n,m,k] = rand(Poisson(Y_NM[n,m-1]))
    #                 end
    #             elseif k == model.K+2
    #                 Y_NMKplus2[n,m,k] = rand(Poisson(pop_N[n]*eps))
    #             else
    #                 Y_NMKplus2[n,m,k] = rand(Poisson(pop_N[n]*alpha*U_NK[n,k] * V_KM[k,m]))
    #             end
    #         end
    #     end
    #     Y_NM[:,m] = dropdims(sum(Y_NMKplus2[:,m,:],dims=2),dims=2)
    # end
    for m in 1:model.M
        for n in 1:model.N

            if m == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y0_N[n] + pop_N[n]*eps + pop_N[n]*alpha*sum(U_NK[n,:] .* V_KM[:,m])),model.D, model.j))
            else
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + pop_N[n]*eps + pop_N[n]*alpha*sum(U_NK[n,:] .* V_KM[:,m])),model.D, model.j))
            end
        end 
    end
    #Y_NM = dropdims(sum(Y_NMKplus2,dims=3),dims=3)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"], #"R_KTS" => state["R_KTS"],"
     "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y0_N"=>info["Y0_N"])
     #Y_NMKplus2"=>Y_NMKplus2)
    return data, state 
end

function backward_sample(model::covidsimple, data, state, mask=nothing, skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])
    #Y_NMKplus2 = copy(state["Y_NMKplus2"])

    Y0_N = copy(state["Y0_N"])
    pop_N = copy(state["pop_N"])

    alphacontribution_NM = zeros(model.N, model.M)
    Y_NMKplus2 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(model.N * model.M)
        m = div(idx - 1, model.N) + 1
        n = mod(idx - 1, model.N) + 1
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        pop = pop_N[n]
        probvec = pop*U_NK[n,:] .* V_KM[:,m]
        mu = sum(probvec) 

        if !isnothing(mask)
            if mask[n,m] == 1   
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + pop*eps + alpha*mu), model.D, model.j))
                #Y_NM[n,m] = rand(Poisson(Ylast + pop*(eps + alpha * U_NK[n,:] .* V_KMm1[:,m-1])))
            end
        end

        Z = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, model.j, Poisson(Ylast + alpha*mu + pop*eps))
        if Z > 0 
            probvec = vcat(alpha*probvec, Ylast, pop*eps)
            Y_NMKplus2[n, m, :] = rand(Multinomial(Z, probvec / sum(probvec)))
        end
    end

    #update alpha (scale)
    post_shape = model.scaleshape + sum(Y_NMKplus2[:,:,1:model.K])
    post_rate = model.scalerate + model.D*sum((pop_N .* U_NK) * V_KM)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    #update time factors
    # Y_MK = dropdims(sum(Y_NMKplus2,dims=1),dims=1)[:,1:model.K]
    # C1_K = alpha * dropdims(sum(pop_N .* U_NK, dims=1), dims=1)
    # @views for m in 1:(model.M)
    #     @views for k in 1:model.K
    #         post_shape = model.c + Y_MK[m,k]
    #         post_rate = model.d + C1_K[k]
    #         V_KM[k,m] = rand(Gamma(post_shape, 1/post_rate))
    #     end
    # end

    #update time factors
    Y_MK = dropdims(sum(Y_NMKplus2,dims=1),dims=1)[:,1:model.K]
    C1_K = model.D*alpha * dropdims(sum(pop_N .* U_NK, dims=1), dims=1)

    l_KM = zeros(model.K,model.M+1)
    q_KM = zeros(model.K,model.M+1)
    #backward pass
    @views @threads for m in model.M:-1:2
        @views for k in 1:model.K
            q_KM[k,m] = log(1 + (C1_K[k]/model.c) + q_KM[k,m+1])
            temp = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
            l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
        end 
    end
    @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    #forward pass
    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                V_KM[k,m] = rand(Gamma(model.starta + Y_MK[m,k] + l_KM[k,m+1], 1/(model.startb + C1_K[k] + model.c*q_KM[k,m+1])))
                #V_KM[k,m] = rand(Gamma(model.starta + l_KM[k,m+1], 1/(model.startb + model.c*q_KM[k,m+1])))
            else
                V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C1_K[k] + model.c*q_KM[k,m+1])))
            end 
        end 
    end 

    #update county factors
    Y_NK = dropdims(sum(Y_NMKplus2,dims=2),dims=2)[:,1:model.K]
    C2_K = model.D * alpha * dropdims(sum(V_KM, dims=2), dims=2)
    @views for n in 1:model.N
        pop =  pop_N[n]
        @views for k in 1:model.K
            post_shape = model.a + Y_NK[n,k]
            post_rate = model.b + pop * C2_K[k]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    #update noise term
    post_shape = model.g + sum(Y_NMKplus2[:,:,model.K+2])
    post_rate = model.h + model.D*model.M*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))



    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps" => eps, "alpha" => alpha,
    "pop_N"=>info["pop_N"],"Y0_N"=>Y0_N,)
    #"Y_NMKplus2"=>Y_NMKplus2)
    return data, state
end

#forecasting code 

function predict(model::covidsimple, state, info, n, m, Ylast)
    pop_N = info["pop_N"]
    eps = state["eps"]
    alpha = state["alpha"]
    pop = pop_N[n]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    return rand(OrderStatistic(Poisson(Ylast + pop*eps + alpha * pop * sum(U_NK[n,:] .* V_KM[:,m])), model.D, model.j))
end

function predict_x(model::covidsimple, state, info, n, mstart, Ystart, x)
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


function forecast(model::covidsimple, state, data, info, Ti)
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
        mu = Ylast .+ info["pop_N"].*(state["eps"] .+ state["alpha"]*mu_NTi[:,i])
        Y_NTi[:,i] = rand.(OrderStatistic.(Poisson.(mu), model.D, model.j))
        Ylast = copy(Y_NTi[:,i])
    end
    return Y_NTi
end