include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct nba <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
    dist::Function
end

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


#should either be a poisson or negative binomial, if negbinomial, it will have p in it already
#need to change to allow p to be random

function evalulateLogLikelihood(model::nba, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 2
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    if col == 1
        mu = sum(state["U1_TK"][home,:]' * state["V_K"] * state["U2_TK"][away,:])
    else
        mu = sum(state["U1_TK"][away,:]' * state["V_K"] * state["U2_TK"][home,:])
    end
    if isnothing(state["p"])
        if model.D == 1
            return logpdf(model.dist(mu), Y)
        else
            return logpdf(OrderStatistic(mode.dist(mu), model.D, model.j), Y)
        end
    else
        if model.D == 1
            return logpdf(model.dist(mu,1-state["p"]), Y)
        else
            return logpdf(OrderStatistic(model.dist(mu,1-state["p"]), model.D, model.j), Y)
        end
    end
end

function sample_likelihood(model::nba, mu,p=nothing)
    if isnothing(p)
        dist = model.dist(mu)
    else
        dist = model.dist(mu,1-p)
    end
    if D == 1
        return rand(dist)
    else
        return rand(OrderStatistic(dist, model.D, model.j))
    end
end


function sample_prior(model::nba, info=nothing,constantinit=nothing)
    p=nothing
    U1_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    U2_TK = rand(Gamma(model.e, 1/model.f), model.T, model.K)
    V_K = rand(Gamma(model.c, 1/model.d), model.K)
    try
        dist = model.dist(1)
    catch e 
        p = rand(Beta(model.alpha, model.beta))
    end

    @assert !isnothing(info)
    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_K" => V_K, "I_NM"=>info["I_NM"], "p"=>p)
    return state
end

function forward_sample(model::nba; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U1_TK = copy(state["U1_TK"])
    U2_TK = copy(state["U2_TK"])
    V_K = copy(state["V_K"])
    p = state["p"]
    # println("V for: ", state["V_K"][1])
    # println("p for: ", state["p"])
    Y_G2 = zeros(model.N, model.M)
    @assert model.M <= 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]

    Y_NM = zeros(model.N,model.M)
    Mu_TT = U1_TK * Diagonal(V_K) * U2_TK'
    @views for n in 1:model.N
        Y_NM[n,1] = sample_likelihood(model, Mu_TT[home_N[n], away_N[n]],p)
        if model.M == 2
            Y_NM[n,2] = sample_likelihood(model, Mu_TT[away_N[n], home_N[n]],p)
        end
    end
    data = Dict("Y_NM" => Int.(Y_NM))
    return data, state
end

function backward_sample(model::nba, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U1_TK = copy(state["U1_TK"])
    U2_TK = copy(state["U2_TK"])
    V_K = copy(state["V_K"])
    I_NM = copy(state["I_NM"])
    p = state["p"]

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 2
    

    Y_NMK = zeros(model.N,model.M,model.K)

    PHI1_TK = zeros(model.T,model.K)
    TAU1_TK = zeros(model.T,model.K)
    PHI2_TK = zeros(model.T,model.K)
    TAU2_TK = zeros(model.T,model.K)
    PHI3_K = zeros(model.K)
    TAU3_K = zeros(model.K)

    Z_NM = zeros(model.N,model.M)
    mu_NM = zeros(model.N,model.M)

    if isnothing(p)
        lik = model.dist
    else
        lik = x -> model.dist(x,1-p)
    end
    
    MU_TT = U1_TK * Diagonal(V_K) * U2_TK'

    # Loop over the non-zeros in Y_NM and allocate
    @views for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        if !isnothing(mask)
            if mask[n,1] == 1 Y_NM[n,1] = sample_likelihood(model, MU_TT[home,away],p) end
            if mask[n,2] == 1 Y_NM[n,2] = sample_likelihood(model, MU_TT[away,home],p) end
        end
        if Y_NM[n, 1] > 0 || model.D != model.j 
            mu = MU_TT[home, away]
            Z = sampleSumGivenOrderStatistic(Y_NM[n, 1], model.D, model.j, lik(mu))
            if lik(mu) isa NegativeBinomial
                Z_NM[n,1] = Z
                Z = sampleCRT(Z, model.D*mu)
            end
            P_K = U1_TK[home, :] .* U2_TK[away, :] .* V_K
            Y_NMK[n, 1, :] = rand(Multinomial(Z, P_K / sum(P_K)))
        end
        if Y_NM[n, 2] > 0 || model.D != model.j
            mu = MU_TT[away, home]
            Z = sampleSumGivenOrderStatistic(Y_NM[n, 2], model.D, model.j, lik(mu))
            if lik(mu) isa NegativeBinomial
                Z_NM[n,2] = Z
                Z = sampleCRT(Z, model.D*mu)
            end
            P_K = U1_TK[away, :] .* U2_TK[home, :] .* V_K
            Y_NMK[n, 2, :] = rand(Multinomial(Z, P_K / sum(P_K)))
        end
    end



    #update p if necessary
    if lik(1) isa NegativeBinomial

        mu_all = 0
        @views for n in 1:model.N
            home = home_N[n]
            away = away_N[n]
            mu_all += MU_TT[home,away] + MU_TT[away,home]
        end

        post_alpha = model.alpha + sum(Z_NM)
        post_beta = model.beta + model.D*mu_all
        p2 = copy(rand(Beta(post_alpha,post_beta)))
        rate_factor = model.D*log(1/(1-p2))
    else
        p2 = nothing
        rate_factor = model.D
    end


    @views for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        #update offense rates and shapes
        PHI1_TK[home,:] += U2_TK[away,:] .* V_K
        TAU1_TK[home,:] += Y_NMK[n,1,:]
        PHI1_TK[away,:] += U2_TK[home,:] .* V_K
        TAU1_TK[away,:] += Y_NMK[n,2,:]
    end

    #now update U1_TK
    @views for t in 1:model.T
        @views for k in 1:model.K
            post_shape = model.a + TAU1_TK[t,k]
            post_rate = model.b + rate_factor*PHI1_TK[t,k]
            U1_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    @views for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        #update defensive rates and shapes
        PHI2_TK[away,:] += U1_TK[home,:] .* V_K
        TAU2_TK[away,:] += Y_NMK[n,1,:]
        PHI2_TK[home,:] += U1_TK[away,:] .* V_K
        TAU2_TK[home,:] += Y_NMK[n,2,:]
    end

    @views for t in 1:model.T
        @views for k in 1:model.K
            post_shape = model.e + TAU2_TK[t,k]
            post_rate = model.f + rate_factor*PHI2_TK[t,k]
            U2_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    @views for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        #update diagonal rates and shapes
        PHI3_K[:] += U1_TK[home,:] .* U2_TK[away,:] + U1_TK[away,:] .* U2_TK[home,:]
        TAU3_K[:] += Y_NMK[n, 1, :] + Y_NMK[n, 2, :]
    end

    #update V_K
    @views for k in 1:model.K
        post_shape = model.c + TAU3_K[k]
        post_rate = model.d + rate_factor*PHI3_K[k]
        V_K[k] = rand(Gamma(post_shape, 1/post_rate))
    end


    # println("beep")
    # flush(stdout)
    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_K" => V_K, "I_NM"=>I_NM,"p"=>p2)
    # println("V bac2: ",state["V_K"][1])
    # println("p bac2: ",state["p"])
    return data, state
end


