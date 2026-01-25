include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct covid1Poisson2 <: MatrixMF
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
end

function evalulateLogLikelihood(model::covid1Poisson2, state, data, info, row, col)
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

function sample_prior(model::covid1Poisson2,info=nothing,constantinit=nothing)
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, 
                "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"])
    return state
end

function forward_sample(model::covid1Poisson2; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model,info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(Poisson.(Mu_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::covid1Poisson2, data, state, mask=nothing)
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

    @views for m in 1:model.M
        @views for k in 1:model.K
            post_shape = model.c + sum(Y_NMKplus2[:, m, k])
            post_rate = model.d + alpha * sum(pop_N .* U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps"=>eps, "alpha"=>alpha,
    "Y0_N"=>info["Y0_N"], "pop_N"=>info["pop_N"])
    return data, state
end
