include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct PoissonDiagonal <: MatrixMF
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
end

function evalulateLogLikelihood(model::PoissonDiagonal, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 2
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    if col == 1
        mu = state["U1_TK"][home,:]' * state["V_KK"] * state["U2_TK"][away,:]
    else
        mu = state["U1_TK"][away,:]' * state["V_KK"] * state["U2_TK"][home,:]
    end
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonDiagonal, info=nothing)
    U1_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    U2_TK = rand(Gamma(model.e, 1/model.f), model.T, model.K)
    V_K = rand(Gamma(model.c, 1/model.d), model.K)
    @assert !isnothing(info)
    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_K" => V_K, "I_NM"=>info["I_NM"])
    return state
end

function forward_sample(model::PoissonDiagonal; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U1_TK = state["U1_TK"]
    U2_TK = state["U2_TK"]
    V_K = state["V_K"]
    
    Y_G2 = zeros(model.N, model.M)
    @assert model.M <= 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]
    #Y_NMKK = zeros(model.N, model.M, model.K, model.K)
    Y_NM = zeros(model.N,model.M)
    Mu_TT = U1_TK * Diagonal(V_K) * U2_TK'
    @views for n in 1:model.N
        Y_NM[n,1] = rand(Poisson(Mu_TT[home_N[n], away_N[n]]))
        if model.M == 2
            Y_NM[n,2] = rand(Poisson(Mu_TT[away_N[n], home_N[n]]))
        end
    end
    #Y_NM = Int.(dropdims(sum(Y_NMKK, dims=(3, 4)),dims=(3,4)))
    
    #data = Dict("Y_NM" => Y_NM, "Y_NMKK"=> Y_NMKK)
    data = Dict("Y_NM" => Int.(Y_NM))
    return data, state
end

function backward_sample(model::PoissonDiagonal, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U1_TK = copy(state["U1_TK"])
    U2_TK = copy(state["U2_TK"])
    V_K = copy(state["V_K"])
    I_NM = copy(state["I_NM"])

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 2

    Y_NMK = zeros(model.N,model.M,model.K)
    # Y_NMdotK = zeros(model.N,model.M,model.K)

    PHI1_TK = zeros(model.T,model.K)
    TAU1_TK = zeros(model.T,model.K)
    PHI2_TK = zeros(model.T,model.K)
    TAU2_TK = zeros(model.T,model.K)
    PHI3_K = zeros(model.K)
    TAU3_K = zeros(model.K)
    

    if !isnothing(mask)
        MU_TT = U1_TK * Diagonal(V_K) * U2_TK'
    end

    locker = SpinLock()
    # Loop over the non-zeros in Y_NM and allocate
    @views @threads for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        if !isnothing(mask)
            if mask[n,1] == 1 Y_NM[n,1] = rand(Poisson(MU_TT[home,away])) end
            if mask[n,2] == 1 Y_NM[n,2] = rand(Poisson(MU_TT[away,home])) end
        end
        if Y_NM[n, 1] > 0
            P_K = U1_TK[home, :] .* U2_TK[away, :] .* V_K
            P_K[:] = P_K / sum(P_K)
            Y_NMK[n, 1, :] = rand(Multinomial(Y_NM[n, 1], P_K))
        end
        if Y_NM[n, 2] > 0
            P_K = U1_TK[away, :] .* U2_TK[home, :] .* V_K
            P_K[:] = P_K / sum(P_K)
            Y_NMK[n, 2, :] = rand(Multinomial(Y_NM[n, 2], P_K))
        end

        # lock(locker)
        # #update offense rates and shapes
        # PHI1_TK[home,:] += U2_TK[away,:] .* V_K
        # TAU1_TK[home,:] += Y_NMK[n,1,:]
        # PHI1_TK[away,:] += U2_TK[home,:] .* V_K
        # TAU1_TK[away,:] += Y_NMK[n,2,:]
        
        # #update defensive rates and shapes
        # PHI2_TK[away,:] += U1_TK[home,:] .* V_K
        # TAU2_TK[away,:] += Y_NMK[n,1,:]
        # PHI2_TK[home,:] += U1_TK[away,:] .* V_K
        # TAU2_TK[home,:] += Y_NMK[n,2,:]
        # unlock(locker)

        #update diagonal rates and shapes
        # PHI3_K[:] += U1_TK[home,:] .* U2_TK[away,:] + U1_TK[away,:] .* U2_TK[home,:]
        # TAU3_K[:] += Y_NMK[n, 1, :] + Y_NMK[n, 2, :]
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
            post_rate = model.b + PHI1_TK[t,k]
            U1_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
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
            post_rate = model.f + PHI2_TK[t,k]
            U2_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    @views for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        #update diagonal rates and shapes
        PHI3_K[:] += U1_TK[home,:] .* U2_TK[away,:] + U1_TK[away,:] .* U2_TK[home,:]
        TAU3_K[:] += Y_NMK[n, 1, :] + Y_NMK[n, 2, :]
    end

    #now update V_K
    @views for k in 1:model.K
        post_shape = model.c + TAU3_K[k]
        post_rate = model.d + PHI3_K[k]
        V_K[k] = rand(Gamma(post_shape, 1/post_rate))[1]
    end

    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_K" => V_K, "I_NM"=>I_NM)
    return data, state
end


