include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct PoissonTimeMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::PoissonTimeMF, state, data, info, row, col)
    @assert col != 1
    Ylast = data["Y_NM"][row,col-1]
    Y = data["Y_NM"][row,col]
    pop = state["pop_N"][row]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    return logpdf(Poisson(Ylast + pop*mu), Y)
end

function sample_prior(model::PoissonTimeMF,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    V_KM[:,1] = zeros(model.K)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"])
    return state
end

function forward_sample(model::PoissonTimeMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_N1 = state["Y_N1"]
    pop_N = state["pop_N"]
    Y_NMKplus1 = zeros(Int, model.N, model.M, model.K + 1)
    for n in 1:model.N
        for m in 1:model.M
            for k in 1:(model.K+1) 
                if m == 1 && k != (model.K+1)
                    Y_NMKplus1[n,m,k] = 0
                elseif m == 1 && k == (model.K+1)
                    Y_NMKplus1[n,m,k] = Y_N1[n,1]
                else
                    if k == (model.K+1)
                        Y_NMKplus1[n,m,k] = rand(Poisson(sum(Y_NMKplus1[n,m-1,:])))
                    else
                        Y_NMKplus1[n,m,k] = rand(Poisson(pop_N[n]* state["U_NK"][n,k] * state["V_KM"][k,m]))
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
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"], "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"],  "Y_NMKplus1"=>Y_NMKplus1)
    return data, state 
end

function backward_sample(model::PoissonTimeMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])


    Y_NMKplus1 = zeros(model.N, model.M, model.K + 1)
    if !isnothing(mask)
        Mu_NM = U_NK * V_KM
    end
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(N * (M))
        m = M - div(idx - 1, N)
        n = mod(idx - 1, N) + 1 
        if m == 1
            continue
        end
        @assert m != 1
        if !isnothing(mask)
            if mask[n,m] == 1
                Ylast = Y_NM[n,m-1]
                Y_NM[n,m] = rand(Poisson(Ylast +pop_N[n]*Mu_NM[n,m]))
                #Y_NM[n,m] = rand(Poisson(Ylast + Mu_NM[n,m]))
            end
        end
        if Y_NM[n, m] > 0
            Ylast = Y_NM[n,m-1]
            P_Kplus1 = vcat(pop_N[n] * U_NK[n, :] .* V_KM[:, m], Ylast)
            #P_Kplus1 = vcat(U_NK[n, :] .* V_KM[:, m], Ylast)
            Y_NMKplus1[n, m, :] = rand(Multinomial(Y_NM[n, m], P_Kplus1 / sum(P_Kplus1)))
        end
    end

    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                continue
            end
            post_shape = model.c + sum(Y_NMKplus1[:, m, k])
            post_rate = model.d + dot(pop_N, U_NK[:, k])
            #post_rate = model.d + sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    
    @views for n in 1:model.N
        @views for k in 1:model.K
            @assert Y_NMKplus1[n, 1, k] == 0
            @assert V_KM[k,1] == 0
            post_shape = model.a + sum(Y_NMKplus1[n, :, k])
            post_rate = model.b + pop_N[n]*sum(V_KM[k, :])
            #post_rate = model.b + sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end



    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N)
    return data, state
end