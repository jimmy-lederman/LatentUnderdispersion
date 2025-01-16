include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
# using Base.Threads
using SpecialFunctions

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        probs = [1]
    else
        probs = vcat([1],[R/(R+i-1) for i in 2:Y])
    end
    return sum(rand.(Bernoulli.(probs)))
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
    D::Int64
    j::Int64
end




function sample_prior(model::covidsimple,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)

    V_KMm1 = rand(Gamma(model.c, 1/model.d), model.K, model.M-1)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))

    state = Dict("U_NK" => U_NK, "V_KMm1" => V_KMm1, #"R_KTS" => R_KTS, 
    "eps"=>eps, "alpha" => alpha,
     "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"], "days_M"=>info["days_M"], "state_N"=>info["state_N"])
    return state
end

function forward_sample(model::covidsimple; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    eps = state["eps"]
    Y_N1 = state["Y_N1"]
    pop_N = state["pop_N"]
    # days_M = state["days_M"]
    # state_N = state["state_N"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KMm1 = state["V_KMm1"]
    Y_NMKplus2 = zeros(Int, model.N, model.M, model.K + 2)
    Y_NM = zeros(Int, model.N, model.M)
    Y_NM[:,1] = Y_N1[:,1]
    
    for m in 2:model.M
        for n in 1:model.N
            for k in 1:(model.K+2)
                if k == model.K+1
                    Y_NMKplus2[n,m,k] = rand(Poisson(Y_NM[n,m-1]))
                elseif k == model.K+2
                    Y_NMKplus2[n,m,k] = rand(Poisson(pop_N[n]*eps))
                else
                    Y_NMKplus2[n,m,k] = rand(Poisson(pop_N[n]*alpha*U_NK[n,k] * V_KMm1[k,m-1]))
                end
            end
        end
        Y_NM[:,m] = dropdims(sum(Y_NMKplus2[:,m,:],dims=2),dims=2)
    end
    #Y_NM = dropdims(sum(Y_NMKplus2,dims=3),dims=3)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KMm1" => state["V_KMm1"], #"R_KTS" => state["R_KTS"],"
     "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"], "days_M"=>info["days_M"], "state_N"=>info["state_N"], "Y_NMKplus2"=>Y_NMKplus2)
    return data, state 
end

function backward_sample(model::covidsimple, data, state, mask=nothing, skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KMm1 = copy(state["V_KMm1"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    days_M = copy(state["days_M"])
    state_N = copy(state["state_N"])

    #mylock = ReentrantLock()

    #alphacontribution_NM = zeros(model.N, model.M)
    Y_NMKplus2 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    # @views for idx in 1:(model.N * model.M)
    #     m = model.M - div(idx - 1, model.N)
    #     n = mod(idx - 1, model.N) + 1 
    #     if m == 1
    #         continue
    #     end
    #@views for m in model.M:-1:2
    @views for m in 2:model.M
        @views for n in 1:model.N
            @assert m != 1
            Ylast = Y_NM[n,m-1]
            pop = pop_N[n]
            # probvec  = 
            # mu = sum(probvec) 
            # alphacontribution_NM[n,m] = mu

            # if !isnothing(mask)
            #     if mask[n,m] == 1   
            #         #Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + pop*eps + alpha*mu), model.D, model.j))
            #         Y_NM[n,m] = rand(Poisson(Ylast + pop*(eps + alpha * U_NK[n,:] .* V_KMm1[:,m-1])))
            #     end
            # end

        #Z = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, model.j, Poisson(Ylast + alpha*mu + pop*eps))
            
            probvec = vcat(alpha*pop*U_NK[n,:] .* V_KMm1[:,m-1], Ylast, pop*eps)
            Y_NMKplus2[n, m, :] = rand(Multinomial(Y_NM[n,m], probvec / sum(probvec)))
        end
    end
    #Y_NMKplus2[:,1,:]  


    #update alpha (scale)
    post_shape = model.scaleshape + sum(Y_NMKplus2[:,:,1:model.K])
    post_rate = model.scalerate + sum(pop_N .* U_NK * V_KMm1)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    #update time factors
    Y_MK = dropdims(sum(Y_NMKplus2,dims=1),dims=1)[:,1:model.K]
    C_K1 = alpha * dropdims(sum(pop_N .* U_NK, dims=1), dims=1)
    @views for m in 1:(model.M-1)
        @views for k in 1:model.K
            post_shape = model.c + Y_MK[m+1,k]
            post_rate = model.d + C_K1[k]
            V_KMm1[k,m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    #update county factors
    Y_NK = dropdims(sum(Y_NMKplus2,dims=2),dims=2)[:,1:model.K]
    C_K2 = alpha * dropdims(sum(V_KMm1, dims=2), dims=2)
    @views for n in 1:model.N
        pop =  pop_N[n]
        @views for k in 1:model.K
            post_shape = model.a + Y_NK[n,k]
            post_rate = model.b + pop * C_K2[k]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    #update noise term
    post_shape = model.g + sum(Y_NMKplus2[:,:,model.K+2])
    post_rate = model.h + (model.M-1)*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))



    state = Dict("U_NK" => U_NK, "V_KMm1" => V_KMm1, "Y_N1"=>Y_N1, #"R_KTS"=>R_KTS, 
    "eps" => eps, "alpha" => alpha,
     "Y_NMKplus2"=>Y_NMKplus2, "pop_N"=>pop_N, "days_M"=>days_M, "state_N" => state_N)
    return data, state
end