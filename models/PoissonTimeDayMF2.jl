include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct PoissonTimeDayMF <: MatrixMF
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
end

function evalulateLogLikelihood(model::PoissonTimeDayMF, state, data, info, row, col)
    @assert col != 1
    Ylast = data["Y_NM"][row,col-1]
    Y = data["Y_NM"][row,col]
    pop = state["pop_N"][row]
    day = state["days_M"][col]
    s = state["state_N"][row]

    eps = state["eps"]
    alpha = state["alpha"]
    mu = sum(state["U_NK"][row,:] .* state["V_KM"][:,col] .* state["R_KTS"][:,day,s])
    return logpdf(Poisson(Ylast + alpha*pop*mu + pop*eps), Y)
end

function sample_prior(model::PoissonTimeDayMF,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    R_KTS = rand(Gamma(model.e, 1/model.f), model.K, model.T, model.S)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    V_KM[:,1] = zeros(model.K)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "R_KTS" => R_KTS, "eps" => eps, "alpha" => alpha,
     "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"], "days_M"=>info["days_M"], "state_N"=>info["state_N"])
    return state
end

function forward_sample(model::PoissonTimeDayMF; state=nothing, info=nothing)
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
    Y_NMKplus1 = zeros(Int, model.N, model.M, model.K + 2)
    @views for n in 1:model.N
        @views for m in 1:model.M
            @views for k in 1:(model.K+2) 
                if m == 1 && k != (model.K+1)
                    Y_NMKplus1[n,m,k] = 0
                elseif m == 1 && k == (model.K+1)
                    Y_NMKplus1[n,m,k] = Y_N1[n,1]
                else
                    if k == (model.K+1)
                        Y_NMKplus1[n,m,k] = rand(Poisson(sum(Y_NMKplus1[n,m-1,:])))
                    elseif k ==  (model.K+2)
                        Y_NMKplus1[n,m,k] = rand(Poisson(pop_N[n]*eps))
                    else
                        day = days_M[m]
                        s = state_N[n]
                        Y_NMKplus1[n,m,k] = rand(Poisson(alpha*pop_N[n]* state["R_KTS"][k,day,s]* state["U_NK"][n,k] * state["V_KM"][k,m]))
                        #Y_NMKplus1[n,m,k] = rand(Poisson(state["U_NK"][n,k] * state["V_KM"][k,m]))
                    end
                end
            end
        end
    end
    Y_NMKplus1[:, 1, model.K+1] = zeros(model.N)
    Y_NM = sum(Y_NMKplus1, dims=3)
    Y_NM[:,1] = Y_N1[:,1]
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"], "R_KTS" => state["R_KTS"], "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"], "days_M"=>info["days_M"], "state_N"=>info["state_N"], "Y_NMKplus1"=>Y_NMKplus1)
    return data, state 
end

function backward_sample(model::PoissonTimeDayMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    R_KTS = copy(state["R_KTS"])
    eps = copy(state["eps"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    days_M = copy(state["days_M"])
    state_N = copy(state["state_N"])
    alpha = state["alpha"]


    Y_NMKplus1 = zeros(model.N, model.M, model.K + 2)
    total_all = 0
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(N * (M))
        m = M - div(idx - 1, N)
        n = mod(idx - 1, N) + 1 
        if m == 1
            continue
        end
        beep = 0
        @assert m != 1
        Ylast = Y_NM[n,m-1]
        pop = pop_N[n]
        day = days_M[m]
        s = state_N[n]
        if !isnothing(mask)
            if mask[n,m] == 1
                probvec  = pop * R_KTS[:,day,s] .* U_NK[n,:] .* V_KM[:,m]
                beep = sum(probvec)
                probvec = alpha*probvec
                Y_NM[n,m] = rand(Poisson(Ylast + sum(probvec)))
            end
        end
        if Y_NM[n, m] > 0
            if !isnothing(mask) && mask[n,m] == 1
                probvec = vcat(probvec, Ylast, pop*eps)
            else
                part1 = pop * R_KTS[:,day,s] .* U_NK[n, :] .* V_KM[:, m]
                probvec = vcat(alpha * part1, Ylast, pop*eps)
                beep = sum(part1)
            end
            Y_NMKplus1[n, m, :] = rand(Multinomial(Y_NM[n, m], probvec / sum(probvec)))
        end
        total_all += beep
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
    post_rate = model.scalerate + temp
    alpha = rand(Gamma(post_shape, 1/post_rate))



    @views for m in 1:model.M
        day = days_M[m]
        @views for k in 1:model.K
            if m == 1
                continue
            end
            post_shape = model.c + sum(Y_NMKplus1[:, m, k])
            post_rate = model.d + alpha * sum(pop_N .* U_NK[:, k] .* R_KTS[k,day,state_N])
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
    #         #println(length(popsubset_A), size(Usubset_AK * Vsubset_KB))
    #         post_rate = model.f + sum(popsubset_A .* (Usubset_AK * Vsubset_KB))
    #         #post_rate = model.f + model.D*dot(popsubset_A, Usubset_AK) * sum(Vsubset_KB[k,:])
    #         R_TS[t, s] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end

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
                post_rate = model.f + alpha * dot(popsubset_A, Usubset_AK[:,k]) * sum(Vsubset_KB[k,:])
                R_KTS[k,t, s] = rand(Gamma(post_shape, 1/post_rate))[1]
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
            post_rate = model.b + alpha * pop * dot(V_KM[k, :],R_KTS[k,days_M,s])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    post_shape = model.g + sum(Y_NMKplus1[:,:,model.K+2])
    post_rate = model.h + model.M*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "R_KTS"=>R_KTS, "eps" => eps, "alpha" => alpha,
     "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N, "days_M"=>days_M, "state_N" => state_N)
    return data, state
end