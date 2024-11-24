include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

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
    starta::Float64
    startb::Float64
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
    pop = info["pop_N"][row]
    day = info["days_M"][col]
    s = info["state_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    mu = sum(state["U_NK"][row,:] .* state["V_KM"][:,col] .* state["R_KTS"][:,day,s])
    return logpdf(Poisson(Ylast + alpha*pop*mu + pop*eps), Y)
end

function forecast(model::PoissonTimeDayMF, state, data, info, Ti)
    lastgamma = state["V_KM"][:,end]
    forecastGamma_KTi = zeros(model.K, Ti)
    for i in 1:Ti
        forecastGamma_KTi[:,i] = rand.(Gamma.(model.c*lastgamma,1/model.c))
        lastgamma = copy(forecastGamma_KTi[:,i])
    end 
    Ylast = data["Y_NM"][:,end]
    #generate new dats of the week
    last_day_of_week = info["days_M"][end]
    days = [mod1(last_day_of_week + i - 1, 7) for i in 1:Ti]

    R_NKTi = permutedims(state["R_KTS"][:,days,info["state_N"]],(3,1,2))
    temp_KTiN = permutedims(state["U_NK"] .* R_NKTi,(2,3,1))

    mu_NTi = permutedims(dropdims(sum(forecastGamma_KTi .* temp_KTiN, dims=1),dims=1),(2,1))

    Y_NTi = zeros(model.N,Ti)
    for i in 1:Ti
        mu = Ylast .+ info["pop_N"].*(state["eps"] .+ state["alpha"]*mu_NTi[:,i])
        Y_NTi[:,i] = rand.(Poisson.(mu))
        Ylast = copy(Y_NTi[:,i])
    end
    return Y_NTi
end

function predict(model::PoissonTimeDayMF, state, info, n, m, Ylast)
    pop_N = info["pop_N"]
    days_M = info["days_M"]
    state_N = info["state_N"]
    eps = state["eps"]
    alpha = state["alpha"]
    day = days_M[m]
    s = state_N[n]
    pop = pop_N[n]
    return rand(Poisson(Ylast + pop*eps + alpha * pop * sum(state["R_KTS"][:,day,s] .* state["U_NK"][n,:] .* state["V_KM"][:,m])))
end

function predict_x(model::PoissonTimeDayMF, state, info, n, mstart, Ystart, x)
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

function sample_prior(model::PoissonTimeDayMF,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    
    pass = false
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.starta,1/model.startb))
                    #V_KM[k,m] = 0
                else
                    #println(k, " ", m, " ", model.c, " ", V_KM[k,m-1])
                    #V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1] + model.d,1/model.c))
                    V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1],1/model.c))
                end
            end
        end
    end

    # V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    #V_KM[:,1] = zeros(model.K)
    R_KTS = rand(Gamma(model.e, 1/model.f), model.K, model.T, model.S)

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

function backward_sample(model::PoissonTimeDayMF, data, state, mask=nothing,skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    R_KTS = copy(state["R_KTS"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])
    Y_NMKplus1 = copy(state["Y_NMKplus1"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    days_M = copy(state["days_M"])
    state_N = copy(state["state_N"])
    # println(V_KM)

    #Y_NMKplus1 = zeros(model.N, model.M, model.K + 2)
    alphacontribution_NM = zeros(model.N, model.M)
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
        probvec = pop * R_KTS[:,day,s] .* U_NK[n,:] .* V_KM[:,m]
        alphacontribution_NM[n,m] = sum(probvec)
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(Poisson(Ylast + pop*eps + sum(alpha*probvec)))
            end
        end
        if Y_NM[n, m] > 0
            probvec = vcat(alpha*probvec, Ylast, pop*eps)
            Y_NMKplus1[n, m, :] = rand(Multinomial(Y_NM[n, m], probvec / sum(probvec)))
        end
    end
    @assert sum(Y_NM[:,2:end]) == sum(Y_NMKplus1)

    post_shape = model.scaleshape + sum(Y_NMKplus1[:,2:end,1:model.K])
    post_rate = model.scalerate + sum(alphacontribution_NM)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    post_shape = model.g + sum(Y_NMKplus1[:,2:end,model.K+2])
    post_rate = model.h + (model.M-1)*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))

    Y_MK = dropdims(sum(Y_NMKplus1,dims=1),dims=1)[:,1:model.K]
    R_NKM = permutedims(R_KTS[:,days_M,state_N], (3,1,2))
    C_KM = alpha * dropdims(sum(pop_N .* U_NK .* R_NKM, dims=1), dims=1)

    l_KM = zeros(model.K,model.M+1)
    q_KM = zeros(model.K,model.M+1)
    #backward pass
    @views for m in model.M:-1:2
        @views for k in 1:model.K
            q_KM[k,m] = log(1 + (C_KM[k,m]/model.c) + q_KM[k,m+1])
            #l_KM[k,m] = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1])
            temp = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
            l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
        end 
    end
    @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    #forward pass
    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                V_KM[k,m] = rand(Gamma(model.starta + l_KM[k,m+1], 1/(model.startb + model.c*q_KM[k,m+1])))
            else
                V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C_KM[k,m] + model.c*q_KM[k,m+1])))
            end 
        end 
    end 

    Y_NK = dropdims(sum(Y_NMKplus1,dims=2),dims=2)[:,1:model.K]
    R_KMN = R_KTS[:,days_M[2:end],state_N]
    C_KN = alpha * dropdims(sum(V_KM[:,2:end] .* R_KMN, dims=2), dims=2)

    @views for n in 1:model.N
        pop =  pop_N[n]
        s = state_N[n]
        @views for k in 1:model.K
            post_shape = model.a + Y_NK[n,k]
            post_rate = model.b + pop * C_KN[k,n]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end
    
    if isnothing(skipupdate) || !("R_KTS" in skipupdate)
        @views for t in 1:model.T
            indices = days_M .== t #filter to days of week
            indices[1] = 0 # do not include first day
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
                    R_KTS[k,t, s] = rand(Gamma(post_shape, 1/post_rate))
                end
            end
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "R_KTS"=>R_KTS, "eps" => eps, "alpha" => alpha,
     "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N, "days_M"=>days_M, "state_N" => state_N)
    return data, state
end