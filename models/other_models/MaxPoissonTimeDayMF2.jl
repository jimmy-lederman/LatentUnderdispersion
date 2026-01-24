include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct MaxPoissonTimeDayMF <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    S::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64 
    h::Float64
    scaleshape::Float64
    scalerate::Float64

    D::Int64
end

function evalulateLogLikelihood(model::MaxPoissonTimeDayMF, state, data, info, row, col)
    @assert col != 1
    Ylast = data["Y_NM"][row,col-1]
    Y = data["Y_NM"][row,col]
    pop = state["pop_N"][row]
    day = state["days_M"][col]
    s = state["state_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    mu = sum(state["U_NK"][row,:] .* state["V_KM"][:,col] .* state["R_KTS"][:,day,s])
    return logpmfMaxPoisson(Y,Ylast+alpha*pop*mu+pop*eps,model.D)
end

function sample_prior(model::MaxPoissonTimeDayMF,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    R_KTS = rand(Gamma(model.e, 1/model.f), model.K, model.T, model.S)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    V_KM[:,1] = zeros(model.K)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "R_KTS" => R_KTS, "eps"=>eps, "alpha" => alpha,
     "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"], "days_M"=>info["days_M"], "state_N"=>info["state_N"])
    return state
end

function forward_sample(model::MaxPoissonTimeDayMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    #Mu_NM = state["U_NK"] * state["V_KM"]
    eps = state["eps"]
    Y_N1 = state["Y_N1"]
    pop_N = state["pop_N"]
    days_M = state["days_M"]
    state_N = state["state_N"]
    alpha = state["alpha"]
    Y_NMKplus1 = zeros(Int, model.N, model.M, model.K + 2, model.D)
    @views for n in 1:model.N
        @views for m in 1:model.M
            @views for k in 1:(model.K+2) 
                if m == 1 && k != (model.K+1)
                    Y_NMKplus1[n,m,k,:] = zeros(model.D)
                elseif m == 1 && k == (model.K+1)
                    Y_NMKplus1[n,m,k,:] = fill(Y_N1[n,1], model.D)
                else
                    if k == (model.K+1)
                        @views for d in 1:model.D
                            Y_NMKplus1[n,m,k,d] = rand(Poisson(sum(Y_NMKplus1[n,m-1,:,d])))
                        end
                    elseif k ==  (model.K+2)
                        Y_NMKplus1[n,m,k,:] = rand(Poisson(pop_N[n]*eps), model.D)
                    else
                        day = days_M[m]
                        s = state_N[n]
                        Y_NMKplus1[n,m,k,:] = rand(Poisson(alpha*pop_N[n]* state["R_KTS"][k,day,s]* state["U_NK"][n,k] * state["V_KM"][k,m]), model.D)
                        #Y_NMKplus1[n,m,k] = rand(Poisson(state["U_NK"][n,k] * state["V_KM"][k,m]))
                    end
                end
            end
        end
    end
    Y_NMKplus1[:, 1, model.K+1,:] = zeros(model.N, model.D)

    Y_NMD = dropdims(sum(Y_NMKplus1, dims=3),dims=3)
    Y_NM = dropdims(maximum(Y_NMD, dims=3),dims=3)
    # println(size(Y_NM), size(Y_NMD), size(Y_N1))
    Y_NM[:,1] = Y_N1[:,1]
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"], "R_KTS" => state["R_KTS"], "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"], "days_M"=>info["days_M"], "state_N"=>info["state_N"], "Y_NMKplus1"=>Y_NMKplus1)
    return data, state 

end

function backward_sample(model::MaxPoissonTimeDayMF, data, state, mask=nothing, skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    R_KTS = copy(state["R_KTS"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    days_M = copy(state["days_M"])
    state_N = copy(state["state_N"])


    Y_NMKplus1 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(N * (M))
        m = M - div(idx - 1, N)
        n = mod(idx - 1, N) + 1 
        if m == 1
            continue
        end

        @assert m != 1
        Ylast = Y_NM[n,m-1]
        pop = pop_N[n]
        day = days_M[m]
        s = state_N[n]
        probvec  = alpha * pop * R_KTS[:,day,s] .* U_NK[n,:] .* V_KM[:,m]
        mu = sum(probvec)

        if !isnothing(mask)
            if mask[n,m] == 1   
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + pop*eps+ sum(probvec)),model.D,model.D))
            end
        end
        
        if Y_NM[n, m] > 0
            Z = sampleSumGivenMax(Y_NM[n, m], model.D, Poisson(Ylast + mu + pop*eps))

            probvec = vcat(probvec, Ylast, pop*eps)

            Y_NMKplus1[n, m, :] = rand(Multinomial(Z, probvec / sum(probvec)))
        end
    end

    post_shape = model.scaleshape + sum(Y_NMKplus1[:,:,1:model.K])
    # R_KMN = R_KTS[:,days_M,state_N]
    # temp_NKM = U_NK .* permutedims(V_KM .* R_KMN, (3,1,2))
    temp = 0
    @views for m in 1:model.M
        @views for n in 1:model.N
            @views for k in 1:model.K
                temp += pop_N[n]*U_NK[n,k]*V_KM[k,m]*R_KTS[k,days_M[m],state_N[n]]
            end 
        end 
    end
    post_rate = model.scalerate + model.D*temp
    alpha = rand(Gamma(post_shape, 1/post_rate))


    @views for m in 1:model.M
        day = days_M[m]
        @views for k in 1:model.K
            if m == 1
                continue
            end
            post_shape = model.c + sum(Y_NMKplus1[:, m, k])
            post_rate = model.d + model.D * alpha * sum(pop_N .* U_NK[:, k] .* R_KTS[k,day,state_N])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    # @views for t in 1:model.T
    #     indices = days_M .== t #filter to days of week
    #     Vsubset_KB = V_KM[:,indices]
    #     @views for s in 1:model.S
    #         indices2 = state_N .== s #filter to state
    #         popsubset_A = pop_N[indices2]
    #         Usubset_AK = U_NK[indices2, :]
    #         Ysubset_ABK = Y_NMKplus1[indices2, indices, :]
            
    #         post_shape = model.e + sum(Ysubset_ABK[:,:,1:model.K])
    #         post_rate = model.f + model.D*sum(popsubset_A .* (Usubset_AK * Vsubset_KB))
    #         #post_rate = model.f + model.D*dot(popsubset_A, Usubset_AK) * sum(Vsubset_KB[k,:])
    #         R_KTS[k,t, s] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end
    
    if isnothing(skipupdate) || !("R_KTS" in skipupdate)
        @views for t in 1:model.T
            indices = days_M .== t #filter to days of week
            Vsubset_KB = V_KM[:,indices]
            @views for s in 1:model.S
                indices2 = state_N .== s #filter to state
                popsubset_A = pop_N[indices2]
                Usubset_AK = U_NK[indices2, :]
                Ysubset_ABK = Y_NMKplus1[indices2, indices, :]
                @views for k in 1:model.K
                    post_shape = model.e + sum(Ysubset_ABK[:,:,k])
                    #println(length(dot(popsubset_A, Usubset_AK[:,k])), " ", length(sum(Vsubset_KB[k,:])))
                    post_rate = model.f + model.D*alpha*dot(popsubset_A, Usubset_AK[:,k]) * sum(Vsubset_KB[k,:])
                    R_KTS[k,t, s] = rand(Gamma(post_shape, 1/post_rate))[1]
                end
            end
        end
    end
    
    @views for n in 1:model.N
        pop =  pop_N[n]
        s = state_N[n]
        @views for k in 1:model.K
            @assert Y_NMKplus1[n, 1, k] == 0
            @assert V_KM[k,1] == 0
            post_shape = model.a + sum(Y_NMKplus1[n, :, k])
            post_rate = model.b + model.D*alpha*pop*dot(V_KM[k, :],R_KTS[k,days_M,s])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    post_shape = model.g + sum(Y_NMKplus1[:,:,model.K+2])
    post_rate = model.h + model.D*model.M*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))

    # post_shape = model.scaleshape + sum(Y_NMKplus1[:,:,1:model.K])
    # R_KMN= R_KTS[:,days_M,state_N]
    # temp_NKM = U_NK .* permutedims(V_KM .* R_KMN, (3,1,2))
    # post_rate = model.scalerate + model.D*sum(temp_NKM)
    # alpha = rand(Gamma(post_shape, 1/post_rate))



    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "R_KTS"=>R_KTS, "eps" => eps, "alpha" => alpha,
     "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N, "days_M"=>days_M, "state_N" => state_N)
    return data, state
end