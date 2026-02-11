include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct covid3Poisson <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    c::Float64
    d::Float64
    g::Float64
    h::Float64
    shapescale::Float64
    shaperate::Float64
    start_V1::Float64
    start_V2::Float64
end

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

function sampleCRTlecam(Y,R,tol=.4)
    Y_max = R * (1/tol - 1)
    if Y <= Y_max || Y <= 100
        return sampleCRT(Y, R)
    else
        out = sampleCRT(Y_max, R)
        mu = R * (polygamma(0, R + Y) - polygamma(0, R + Y_max))
        return out + rand(Poisson(mu))
    end
end

function evalulateLogLikelihood(model::covid3Poisson, state, data, info, row, col)
    @assert !isnothing(info)
        if col == 1
        Ylast = info["Y0_N"][row]
    else
        Ylast = data["Y_NM"][row,col-1]
    end
    Y = data["Y_NM"][row,col]
    D = state["D_NM"][row,col]
    pop = info["pop_N"][row]
    eps = state["eps"]
    alpha = state["alpha"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+alpha*pop*mu+pop*eps
    return logpdf(Poisson(rate), Y)
end

function sample_prior(model::covid3Poisson,info=nothing,constantinit=nothing)
    pass = false
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.start_V1,1/model.start_V2))
                    #V_KM[k,m] = rand(Gamma(model.c*model.start_V+model.d,1/model.c))
                else
                    V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1]+model.d,1/model.c))
                end
            end
        end
    end
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, 
                "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"])
    return state
end

function forward_sample(model::covid3Poisson; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(Poisson.(Mu_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid3Poisson, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Y0_N = copy(state["Y0_N"])
    pop_N = copy(state["pop_N"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])
    Y_NMKplus2 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for idx in 1:(model.N * model.M)
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1 
        if m == 1
            Ylast = Y0_N[n]
        else
            Ylast = Y_NM[n,m-1]
        end
        pop = pop_N[n]
        probvec = pop*U_NK[n,:] .* V_KM[:,m]
        mu1 = sum(probvec) 
        mu = Ylast + pop*eps + alpha*mu1
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(Poisson(mu))
            end
        end
        if Y_NM[n, m] > 0
            P_K = vcat(alpha*probvec, Ylast, pop*eps)
            y_k = rand(Multinomial(Y_NM[n, m],  P_K / sum(P_K)))
            Y_NMKplus2[n, m, :] = y_k
        end
    end
    @assert sum(Y_NMK) == sum(Y_NM)

    #update alpha (scale)
    post_shape = model.scaleshape + sum(Y_NMKplus2[:,:,1:model.K])
    post_rate = model.scalerate + sum((pop_N .* U_NK) * V_KM .* D_NM)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    #update noise term
    post_shape = model.g + sum(Y_NMKplus2[:,:,model.K+2])
    post_rate = model.h + sum(D_NM' * pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))
    
    Y_NK = dropdims(sum(Y_NMKplus2, dims = 2), dims = 2)
    @views for k in 1:model.K
         U_NK[:, k] = rand(Dirichlet(fill(model.a, model.N) .+ Y_NK[:,k]))
    end

    #update time factors
    Y_MK = dropdims(sum(Y_NMKplus2,dims=1),dims=1)[:,1:model.K]
    C1_KM = alpha * (pop_N .* U_NK)' * D_NM

    l_KM = zeros(model.K,model.M+1)
    q_KM = zeros(model.K,model.M+1)
    
    #backward pass
    @views for k in 1:model.K
        @views for m in model.M:-1:2
            q_KM[k,m] = log(1 + (C1_KM[k,m]/model.c) + q_KM[k,m+1])
            
            temp = sampleCRTlecam(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
            
            l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
        end 
    end
    
    @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    #forward pass
    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                V_KM[k,m] = rand(Gamma(model.start_V1 + Y_MK[m,k] + l_KM[k,m+1], 1/(model.start_V2 + C1_KM[k,m] + model.c*q_KM[k,m+1])))
            else
                V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C1_KM[k,m] + model.c*q_KM[k,m+1])))
            end 
        end 
    end 

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"])
    return data, state
end
