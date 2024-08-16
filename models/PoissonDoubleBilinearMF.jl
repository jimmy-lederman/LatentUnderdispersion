include("MatrixMF.jl")
using Distributions
using LinearAlgebra

struct PoissonDoubleBilinear <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::PoissonDoubleBilinear, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 2
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    if col == 1
        mu = state["U_TK"][home,:]' * state["V_KK"] * state["U_TK"][away,:]
    else
        mu = state["U_TK"][away,:]' * state["V_KK"] * state["U_TK"][home,:]
    end
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonDoubleBilinear, info=nothing)
    U_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    V_KK = rand(Gamma(model.c, 1/model.d), model.K, model.K)
    @assert !isnothing(info)
    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>info["I_NM"])
    return state
end

function forward_sample(model::PoissonDoubleBilinear; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U_TK = state["U_TK"]
    V_KK = state["V_KK"]
    
    Y_G2 = zeros(model.N, model.M)
    @assert model.M <= 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]
    #Y_NMKK = zeros(model.N, model.M, model.K, model.K)
    Y_NM = zeros(model.N,model.M)
    Mu_TT = U_TK * V_KK * U_TK'
    for n in 1:model.N
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

function backward_sample(model::PoissonDoubleBilinear, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_TK = copy(state["U_TK"])
    V_KK = copy(state["V_KK"])
    I_NM = copy(state["I_NM"])

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 2
    Y_NMKdot = zeros(model.N,model.M,model.K)
    Y_NMdotK = zeros(model.N,model.M,model.K)
    Y1_dotKKdot = zeros(model.K, model.K)
    P1_KtimesK = zeros(model.K*model.K)
    Y2_dotKKdot = zeros(model.K, model.K)
    P2_KtimesK = zeros(model.K*model.K)

    PHI_tk = zeros(model.T, model.K)
    if !isnothing(mask)
        MU_TT = U_TK * V_KK * U_TK'
    end
    # Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        @assert home != away
        if !isnothing(mask)
            if mask[n,1] == 1 Y_NM[n,1] = rand(Poisson(MU_TT[home,away])) end
            if mask[n,2] == 1 Y_NM[n,2] = rand(Poisson(MU_TT[away,home])) end
        end
        if Y_NM[n, 1] > 0
            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U_TK[away,:])
                P_K[k] =  U_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Y_NMKdot[n,1,:] = rand(Multinomial(Y_NM[n, 1], P_K))
            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U_TK[home,:])
                P_K[k] =  U_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Y_NMdotK[n,1,:] = rand(Multinomial(Y_NM[n, 1], P_K))
        end
        if Y_NM[n, 2] > 0
            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U_TK[home,:])
                #PHI_tk[away,k] += temp
                P_K[k] = U_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Y_NMKdot[n,2,:] = rand(Multinomial(Y_NM[n, 2], P_K))
            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U_TK[away,:])
                #PHI_tk[home,k] += temp
                P_K[k] =  U_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Y_NMdotK[n,2,:] = rand(Multinomial(Y_NM[n, 2], P_K))
        end
    end

    #and then thin the total across the K^2 subcategories, but, pivotally,
    # we only need to think across K^2 once per iteration, not one per data point
    Y1 = sum(Y_NM[:,1])
    Y2 = sum(Y_NM[:,2])
    k = 1
    Q1_KK = zeros(model.K,model.K)
    Q2_KK = zeros(model.K,model.K)
    for k1 in 1:model.K
        for k2 in 1:model.K
            for n in 1:model.N
                home = home_N[n]
                away = away_N[n]
                Q1_KK[k1,k2] += U_TK[home,k1]*U_TK[away,k2] #keep track of these to use later
                Q2_KK[k1,k2] += U_TK[away,k1]*U_TK[home,k2]
            end
            P1_KtimesK[k] += V_KK[k1,k2]*Q1_KK[k1,k2]
            P2_KtimesK[k] += V_KK[k1,k2]*Q2_KK[k1,k2]
            k+=1
        end
    end 
    P1_KtimesK /= sum(P1_KtimesK)
    P2_KtimesK /= sum(P2_KtimesK)
    #sample vector of length K*K
    vec1_KtimesK = rand(Multinomial(Y1, P1_KtimesK))
    vec2_KtimesK = rand(Multinomial(Y2, P2_KtimesK))
    #reshape into K1 by K2 matrix
    Y1_dotKKdot[:,:] = permutedims(reshape(vec1_KtimesK, model.K, model.K))
    Y2_dotKKdot[:,:] = permutedims(reshape(vec2_KtimesK, model.K, model.K))

    #update the model parameters
    #first update V_KK
    for k1 in 1:model.K
        for k2 in 1:model.K
            post_shape = model.c + Y1_dotKKdot[k1,k2] + Y2_dotKKdot[k1,k2]
            post_rate = model.d + Q1_KK[k1,k2] + Q2_KK[k1,k2]
            V_KK[k1,k2] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    #calculate paramters to update U_TK
    PHI_TK = zeros(model.T,model.K)
    TAU_TK = zeros(model.T,model.K)
    for n in 1:model.N
        for k in 1:model.K
            home = home_N[n]
            away = away_N[n]
            PHI_TK[away,k] += dot(U_TK[home,:],(V_KK[:,k] + V_KK[k,:]))
            PHI_TK[home,k] += dot(U_TK[away,:],(V_KK[k,:] + V_KK[:,k]))
            TAU_TK[away,k] += Y_NMdotK[n,1,k] + Y_NMKdot[n,2,k]
            TAU_TK[home,k] += Y_NMKdot[n,1,k] + Y_NMdotK[n,2,k]
        end
    end

    #now update U_TK
    for t in 1:model.T
        for k in 1:model.K
            post_shape = model.a + TAU_TK[t,k]
            post_rate = model.b + PHI_TK[t,k]
            U_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>I_NM)
    return data, state
end


