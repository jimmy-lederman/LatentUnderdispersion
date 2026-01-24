include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct MaxPoissonTimeMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64
    h::Float64
    D::Int64
end

function evalulateLogLikelihood(model::MaxPoissonTimeMF, state, data, info, row, col)
    @assert col != 1
    Ylast = data["Y_NM"][row,col-1]
    Y = data["Y_NM"][row,col]
    pop = state["pop_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    mu = sum(state["U_NK"][row,:] .* state["V_KM"][:,col])
    return logpmfMaxPoisson(Y,Ylast+alpha*pop*mu+pop*eps,model.D)
end

function sample_prior(model::MaxPoissonTimeMF,info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    V_KM[:,1] = zeros(model.K)
    alpha = rand(Gamma(model.e, 1/model.f))
    eps = rand(Gamma(model.g, 1/model.h))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps" => eps, "alpha" => alpha,
     "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"], "days_M"=>info["days_M"])
    return state
end

function forward_sample(model::MaxPoissonTimeMF; state=nothing, info=nothing, constantinit=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    #Mu_NM = state["U_NK"] * state["V_KM"]
    eps = state["eps"]
    Y_N1 = state["Y_N1"]
    pop_N = state["pop_N"]
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
                        Y_NMKplus1[n,m,k,:] = rand(Poisson(alpha*pop_N[n]*state["U_NK"][n,k] * state["V_KM"][k,m]), model.D)
                        #Y_NMKplus1[n,m,k] = rand(Poisson(state["U_NK"][n,k] * state["V_KM"][k,m]))
                    end
                end
            end
        end
    end
    Y_NMKplus1[:, 1, model.K+1,:] = zeros(model.N, model.D)

    Y_NM = zeros(Int, model.N, model.M)
    for n in 1:model.N
        for m in 1:model.M
            if m > 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + pop_N[n]*eps + alpha * pop_N[n] * sum(state["U_NK"][n,:] .* state["V_KM"][:,m])), model.D, model.D))
            end
        end
    end

    # Y_NMD = dropdims(sum(Y_NMKplus1, dims=3),dims=3)
    # Y_NM = dropdims(maximum(Y_NMD, dims=3),dims=3)
    # println(size(Y_NM), size(Y_NMD), size(Y_N1))
    Y_NM[:,1] = Y_N1[:,1]
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"],  "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"], "Y_NMKplus1"=>Y_NMKplus1)
    return data, state 

end

function backward_sample(model::MaxPoissonTimeMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    eps = copy(state["eps"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    alpha = copy(state["alpha"])


    Y_NMKplus1 = zeros(model.N, model.M, model.K + 2)
    alphacontribution_NM = zeros(model.N, model.M)
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
        probvec = pop * U_NK[n,:] .* V_KM[:,m]
        alphacontribution_NM[n,m] = sum(probvec)
        if !isnothing(mask)
            if mask[n,m] == 1
                Ylast = Y_NM[n,m-1]
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + pop*eps+ alpha*sum(probvec)),model.D,model.D))
            end
        end
        if Y_NM[n, m] > 0
            Z = sampleSumGivenMax(Y_NM[n, m], model.D, Poisson(Ylast + alpha*sum(probvec) + pop*eps))
            probvec = vcat(alpha*probvec, Ylast, pop*eps)
            #P_Kplus1 = vcat(U_NK[n, :] .* V_KM[:, m], Ylast)
            Y_NMKplus1[n, m, :] = rand(Multinomial(Z, probvec / sum(probvec)))
        end
    end

    #update scale
    post_shape = model.e + sum(Y_NMKplus1[:,2:end,1:model.K])
    post_rate = model.f + model.D*sum(alphacontribution_NM)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                continue
            end
            post_shape = model.c + sum(Y_NMKplus1[:, m, k])
            post_rate = model.d + model.D*alpha*dot(pop_N, U_NK[:, k])
            #post_rate = model.d + sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
        end
    end
    
    @views for n in 1:model.N
        @views for k in 1:model.K
            @assert Y_NMKplus1[n, 1, k] == 0
            @assert V_KM[k,1] == 0
            post_shape = model.a + sum(Y_NMKplus1[n, :, k])
            post_rate = model.b + model.D*alpha*pop_N[n]*sum(V_KM[k, :])
            #post_rate = model.b + sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    #update noise
    post_shape = model.g + sum(Y_NMKplus1[:,:,model.K+2])
    post_rate = model.h + model.D*model.M*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))



    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N, "eps"=>eps, "alpha"=>alpha)
    return data, state
end