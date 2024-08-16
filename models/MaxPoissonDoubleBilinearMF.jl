include("MatrixMF.jl")
include("PoissonMaxFunctions.jl")
using Distributions

struct MaxPoissonDoubleBilinear <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    D::Int64
end



function evalulateLogLikelihood(model::MaxPoissonDoubleBilinear, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 2
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    if col == 1
        mu = state["U_TK"][home,:]' * state["V_KK"] * state["U_TK"][away,:]
    else
        mu = state["U_TK"][away,:]' * state["V_KK"] * state["U_TK"][home,:]
    end
    logpmfMaxPoisson(Y,mu,model.D)
end

function sample_prior(model::MaxPoissonDoubleBilinear, info=nothing)
    U_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    V_KK = rand(Gamma(model.c, 1/model.d), model.K, model.K)
    @assert !isnothing(info)
    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>info["I_NM"])
    return state
end

function forward_sample(model::MaxPoissonDoubleBilinear; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U_TK = state["U_TK"]
    V_KK = state["V_KK"]
    
    Y_G2 = zeros(model.N, model.M)
    @assert model.M == 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]
    Y_NM = zeros(Int, model.N,model.M)
    Mu_TT = U_TK * V_KK * U_TK'
    for n in 1:model.N
        Y_NM[n,1] = rand(OrderStatistic(Poisson(Mu_TT[home_N[n], away_N[n]]), model.D, model.D))
        if model.M == 2
            Y_NM[n,2] = rand(OrderStatistic(Poisson(Mu_TT[away_N[n], home_N[n]]), model.D, model.D))
        end
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

function backward_sample(model::MaxPoissonDoubleBilinear, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_TK = copy(state["U_TK"])
    V_KK = copy(state["V_KK"])
    I_NM = copy(state["I_NM"])

    Z_NMKdot = zeros(model.N,model.M,model.K)
    Z_NMdotK = zeros(model.N,model.M,model.K)
    Z1_dotKKdot = zeros(model.K, model.K)
    P1_KtimesK = zeros(model.K*model.K)
    Z2_dotKKdot = zeros(model.K, model.K)
    P2_KtimesK = zeros(model.K*model.K)

    Z_NM = zeros(Int, model.N,model.M)

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M <= 2
    
    MU_TT = U_TK * V_KK * U_TK'

    # Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        @assert home != away
        if !isnothing(mask)
            if mask[n,1] == 1 Y_NM[n,1] = rand(OrderStatistic(Poisson(MU_TT[home,away]),model.D,model.D)) end
            if mask[n,2] == 1 Y_NM[n,2] = rand(OrderStatistic(Poisson(MU_TT[away,home]),model.D,model.D)) end
        end
        if Y_NM[n, 1] > 0

            Z_NM[n, 1] = sampleSumGivenMax(Y_NM[n, 1], model.D, Poisson(MU_TT[home, away]))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U_TK[away,:])
                P_K[k] =  U_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMKdot[n,1,:] = rand(Multinomial(Z_NM[n, 1], P_K))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U_TK[home,:])
                P_K[k] =  U_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMdotK[n,1,:] = rand(Multinomial(Z_NM[n, 1], P_K))
        end
        if Y_NM[n, 2] > 0
            Z_NM[n, 2] = sampleSumGivenMax(Y_NM[n, 2], model.D, Poisson(MU_TT[away, home]))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U_TK[home,:])
                #PHI_tk[away,k] += temp
                P_K[k] = U_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMKdot[n,2,:] = rand(Multinomial(Y_NM[n, 2], P_K))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U_TK[away,:])
                #PHI_tk[home,k] += temp
                P_K[k] =  U_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMdotK[n,2,:] = rand(Multinomial(Y_NM[n, 2], P_K))
        end
    end

    Z1 = sum(Z_NM[:,1])
    Z2 = sum(Z_NM[:,2])
    k = 1
    Q1_KK = zeros(model.K,model.K)
    Q2_KK = zeros(model.K,model.K)
    for k1 in 1:model.K
        for k2 in 1:model.K
            for n in 1:model.N
                home = home_N[n]
                away = away_N[n]
                Q1_KK[k1,k2] += U_TK[home,k1]*U_TK[away,k2] #keep track of these to update V_KK
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
    vec1_KtimesK = rand(Multinomial(Z1, P1_KtimesK))
    vec2_KtimesK = rand(Multinomial(Z2, P2_KtimesK))
    #reshape into K1 by K2 matrix
    Z1_dotKKdot[:,:] = permutedims(reshape(vec1_KtimesK, model.K, model.K))
    Z2_dotKKdot[:,:] = permutedims(reshape(vec2_KtimesK, model.K, model.K))


    for k1 in 1:model.K
        for k2 in 1:model.K
            post_shape = model.c + Z1_dotKKdot[k1,k2] + Z2_dotKKdot[k1,k2]
            post_rate = model.d + model.D*(Q1_KK[k1,k2] + Q2_KK[k1,k2])
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
            TAU_TK[away,k] += Z_NMdotK[n,1,k] + Z_NMKdot[n,2,k]
            TAU_TK[home,k] += Z_NMKdot[n,1,k] + Z_NMdotK[n,2,k]
        end
    end

    #now update U_TK
    for t in 1:model.T
        for k in 1:model.K
            post_shape = model.a + TAU_TK[t,k]
            post_rate = model.b + model.D*PHI_TK[t,k]
            U_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>I_NM)
    return data, state
end




