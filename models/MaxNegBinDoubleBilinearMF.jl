include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
using Distributions
using PyCall
using LogExpFunctions

struct MaxNegBinDoubleBilinear <: MatrixMF
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
    p::Float64
    D::Int64
    
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

function logprobsymbolic(s,z,p,D)
    p_1 = 0
    sympy = pyimport("sympy") 
    s1, z1, p1, p2 = sympy.symbols("s1 z1 p1 p2")
    beep = D*sympy.log(sympy.betainc(s1,z1,p1,p2))
    result1 = beep.subs(s1, Y+1)
    result1 = result1.subs(z1, Int(round(z)))
    result1 = result1.subs(p1, p_1)
    result1 = result1.subs(p2, p)
    result1 = convert(Float64, sympy.N(result1, 50))
    if s == 0
        return result1
    else
        s1, z1, p1, p2 = sympy.symbols("s1 z1 p1 p2")
        beep = D*sympy.log(sympy.betainc(s1,z1,p1,p2))
        result2 = beep.subs(s1, Y)
        result2 = result2.subs(z1, Int(round(z)))
        result2 = result2.subs(p1, p_1)
        result2 = result2.subs(p2, p)
        result2 = convert(Float64, sympy.N(result2, 50))
        return logsubexp(result1,result2)
    end
end

function evalulateLogLikelihood(model::MaxNegBinDoubleBilinear, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 2
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    if col == 1
        mu = state["U1_TK"][home,:]' * state["V_KK"] * state["U2_TK"][away,:]
    else
        mu = state["U1_TK"][away,:]' * state["V_KK"] * state["U2_TK"][home,:]
    end
    #try 1
    try
        llik = logpdf(OrderStatistic(NegativeBinomial(mu,model.p), D, D), Y)
        if isinf(llik)
            llik =  logsubexp(model.D*log1mexp(log(beta_inc(mu,Y+1,model.p,1-model.p)[2])), model.D*log1mexp(log(beta_inc(mu,Y,model.p,1-model.p)[2])))
        end
        return llik
    catch ex
        print("Y: ", Y, " mu: ", mu)
        llik = logprobsymbolic(Y,mu,model.p,model.D)
        return llik
    end
end

function sample_prior(model::MaxNegBinDoubleBilinear, info=nothing)
    U1_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    U2_TK = rand(Gamma(model.e, 1/model.f), model.T, model.K)
    V_KK = rand(Gamma(model.c, 1/model.d), model.K, model.K)
    @assert !isnothing(info)
    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_KK" => V_KK, "I_NM"=>info["I_NM"])
end

function forward_sample(model::MaxNegBinDoubleBilinear; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U1_TK = state["U1_TK"]
    U2_TK = state["U2_TK"]
    V_KK = state["V_KK"]
    
    Y_G2 = zeros(model.N, model.M)
    @assert model.M <= 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]
    #Y_NMKK = zeros(model.N, model.M, model.K, model.K)
    Y_NM = zeros(model.N,model.M)
    Mu_TT = U1_TK * V_KK * U2_TK'
    for n in 1:model.N
        Y_NM[n,1] = rand(OrderStatistic(NegativeBinomial(Mu_TT[home_N[n], away_N[n]],model.p), model.D, model.D))
        Y_NM[n,2] = rand(OrderStatistic(NegativeBinomial(Mu_TT[away_N[n], home_N[n]],model.p), model.D, model.D))
    end
    #Y_NM = Int.(dropdims(sum(Y_NMKK, dims=(3, 4)),dims=(3,4)))
    
    #data = Dict("Y_NM" => Y_NM, "Y_NMKK"=> Y_NMKK)
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

function backward_sample(model::MaxNegBinDoubleBilinear, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U1_TK = copy(state["U1_TK"])
    U2_TK = copy(state["U2_TK"])
    V_KK = copy(state["V_KK"])
    I_NM = copy(state["I_NM"])

    Z_NMKdot = zeros(model.N,model.M,model.K)
    Z_NMdotK = zeros(model.N,model.M,model.K)
    Z1_dotKKdot = zeros(model.K, model.K)
    P1_KtimesK = zeros(model.K*model.K)
    Z2_dotKKdot = zeros(model.K, model.K)
    P2_KtimesK = zeros(model.K*model.K)

    Z1_NM = zeros(Int, model.N,model.M)
    Z2_NM = zeros(Int, model.N,model.M)
    

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 2

    PHI1_TK = zeros(model.T,model.K)
    TAU1_TK = zeros(model.T,model.K)
    PHI2_TK = zeros(model.T,model.K)
    TAU2_TK = zeros(model.T,model.K)
    
    P_KtimesK = zeros(model.K*model.K)
    MU_TT = U1_TK * V_KK * U2_TK'

    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        @assert home != away
        if !isnothing(mask)
            if mask_NM[n,1] == 1 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[home,away],model.p),model.D,model.D)) end
            if mask_NM[n,2] == 1 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[away,home],model.p),model.D,model.D)) end
        end
        if Y_NM[n, 1] > 0
            Z1_NM[n,1] = sampleSumGivenMax(Y_NM[n, 1], model.D, NegativeBinomial(MU_TT[home, away],model.p))
            Z2_NM[n,1] = sampleCRT(Z1_NM[n,1],model.D*MU_TT[home, away])

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U2_TK[away,:])
                P_K[k] =  U1_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMKdot[n,1,:] = rand(Multinomial(Z2_NM[n, 1], P_K))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U1_TK[home,:])
                P_K[k] =  U2_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMdotK[n,1,:] = rand(Multinomial(Z2_NM[n, 1], P_K))
        end
        if Y_NM[n, 2] > 0
            Z1_NM[n,2] = sampleSumGivenMax(Y_NM[n, 2], model.D, NegativeBinomial(MU_TT[away, home],model.p))
            Z2_NM[n,2] = sampleCRT(Z1_NM[n,2],model.D*MU_TT[away, home])

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[k,:],U2_TK[home,:])
                #PHI_tk[away,k] += temp
                P_K[k] = U1_TK[away,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMKdot[n,2,:] = rand(Multinomial(Z2_NM[n, 2], P_K))

            P_K = zeros(model.K)
            for k in 1:model.K
                temp = dot(V_KK[:,k],U1_TK[away,:])
                #PHI_tk[home,k] += temp
                P_K[k] =  U2_TK[home,k]*temp
            end
            P_K /= sum(P_K)
            Z_NMdotK[n,2,:] = rand(Multinomial(Z2_NM[n, 2], P_K))
        end

        for k in 1:model.K
            #update offense rates and shapes
            PHI1_TK[home,k] += dot(U2_TK[away,:],V_KK[k,:])
            TAU1_TK[home,k] += Z_NMKdot[n,1,k]
            PHI1_TK[away,k] += dot(U2_TK[home,:],V_KK[k,:])
            TAU1_TK[away,k] += Z_NMKdot[n,2,k]
            
            #update defensive rates and shapes
            PHI2_TK[away,k] += dot(U1_TK[home,:],V_KK[:,k])
            TAU2_TK[away,k] += Z_NMdotK[n,1,k]
            PHI2_TK[home,k] += dot(U1_TK[away,:],V_KK[:,k])
            TAU2_TK[home,k] += Z_NMdotK[n,2,k]
        end
    end


    #now update U_TK
    for t in 1:model.T
        for k in 1:model.K
            post_shape = model.a + TAU1_TK[t,k]
            post_rate = model.b + model.D*log(1/(model.p))*PHI1_TK[t,k]
            U1_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]

            post_shape = model.e + TAU2_TK[t,k]
            post_rate = model.f + model.D*log(1/(model.p))*PHI2_TK[t,k]
            U2_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    Z1 = sum(Z2_NM[:,1])
    Z2 = sum(Z2_NM[:,2])
    k = 1
    Q1_KK = zeros(model.K,model.K)
    Q2_KK = zeros(model.K,model.K)
    for k1 in 1:model.K
        for k2 in 1:model.K
            for n in 1:model.N
                home = home_N[n]
                away = away_N[n]
                Q1_KK[k1,k2] += U1_TK[home,k1]*U2_TK[away,k2] #keep track of these to update V_KK
                Q2_KK[k1,k2] += U1_TK[away,k1]*U2_TK[home,k2]
            end
            P1_KtimesK[k] += V_KK[k1,k2]*Q1_KK[k1,k2]
            P2_KtimesK[k] += V_KK[k1,k2]*Q2_KK[k1,k2]
            k+=1
        end
    end 
    if Z1 > 0
        P1_KtimesK /= sum(P1_KtimesK)
        #sample vector of length K*K
        vec1_KtimesK = rand(Multinomial(Z1, P1_KtimesK))  
        #reshape into K1 by K2 matrix
        Z1_dotKKdot[:,:] = permutedims(reshape(vec1_KtimesK, model.K, model.K))
    end
    if Z2 > 0
        P2_KtimesK /= sum(P2_KtimesK)
        vec2_KtimesK = rand(Multinomial(Z2, P2_KtimesK))
        
        Z2_dotKKdot[:,:] = permutedims(reshape(vec2_KtimesK, model.K, model.K))
    end


    for k1 in 1:model.K
        for k2 in 1:model.K
            post_shape = model.c + Z1_dotKKdot[k1,k2] + Z2_dotKKdot[k1,k2]
            post_rate = model.d + model.D*log(1/(model.p))*(Q1_KK[k1,k2] + Q2_KK[k1,k2])
            V_KK[k1,k2] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U1_TK" => U1_TK, "U2_TK" => U2_TK, "V_KK" => V_KK, "I_NM"=>I_NM)

    # MU_TT = U1_TK * V_KK * U2_TK'
    # # Loop over the non-zeros in Y_NM and allocate
    # for n in 1:model.N
    #     home = home_N[n]
    #     away = away_N[n]
    #     @assert home != away
    #     for m in 1:model.M
    #         if !isnothing(mask)
    #             if mask[n,m] == 1
    #                 if m == 1 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[home,away],model.p),model.D,model.D)) end
    #                 if m == 2 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[away,home],model.p),model.D,model.D)) end
    #             end
    #         end
    #         if Y_NM[n, m] > 0
    #             #sample latent poissons
    #             if m == 1
    #                 Z1_NM[n,m] = sampleSumGivenMax(Y_NM[n, m], model.D, NegativeBinomial(MU_TT[home, away],model.p))
    #                 Z2_NM[n,m] = sampleCRT(Z1_NM[n,m],model.D*MU_TT[home, away])
    #             else
    #                 Z1_NM[n,m] = sampleSumGivenMax(Y_NM[n, m], model.D, NegativeBinomial(MU_TT[away, home],model.p))
    #                 Z2_NM[n,m] = sampleCRT(Z1_NM[n,m],model.D*MU_TT[away, home])
    #             end
                
    #             if m == 1
    #                 k = 1
    #                 for k1 in 1:K
    #                     for k2 in 1:K
    #                         P_KtimesK[k] = U_TK[home,k1]*V_KK[k1,k2]*U_TK[away,k2]
    #                         k += 1
    #                     end
    #                 end
    #                 P_KtimesK /= sum(P_KtimesK)
    #                 #sample vector of length K*K
    #                 vec_KtimesK = rand(Multinomial(Z2_NM[n, m], P_KtimesK))
    #                 #reshape into K by K matrix
    #                 Z_NMKK[n,m,:,:] = permutedims(reshape(vec_KtimesK, model.K, model.K))
    #             else
    #                 k = 1
    #                 for k1 in 1:K
    #                     for k2 in 1:K
    #                         P_KtimesK[k] = U_TK[away,k1]*V_KK[k1,k2]*U_TK[home,k2]
    #                         k += 1
    #                     end
    #                 end
    #                 P_KtimesK /= sum(P_KtimesK)
    #                 #sample vector of length K*K
    #                 vec_KtimesK = rand(Multinomial(Z2_NM[n, m], P_KtimesK))
    #                 #reshape into K1 by K2 matrix
    #                 Z_NMKK[n,m,:,:] = permutedims(reshape(vec_KtimesK, model.K, model.K))
    #             end
    #         end
    #     end
    # end

    # #update the model parameters
    # #first update V_KK
    # for k1 in 1:model.K
    #     for k2 in 1:model.K
    #         beep_k1k2 = 0
    #         boop_k1k2 = 0
    #         for n in 1:model.N
    #             @assert model.M == 2
    #             beep_k1k2 += Z_NMKK[n,1,k1,k2] + Z_NMKK[n,2,k1,k2]
    #             boop_k1k2 += U_TK[home_N[n],k1]*U_TK[away_N[n],k2] + U_TK[away_N[n],k1]*U_TK[home_N[n],k2]
    #         end
    #         post_shape = model.c + beep_k1k2
    #         post_rate = model.d + model.D*log(1/(model.p))*boop_k1k2
    #         V_KK[k1,k2] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end

    # #now update U_TK
    # for t in 1:model.T
    #     for k in 1:model.K
    #         #calcultate phi_tk and tau_tk
    #         phi_tk = 0
    #         tau_tk = 0
    #         for n in 1:model.N 
    #             if t == away_N[n]
    #                 for k1 in 1:model.K 
    #                     phi_tk += U_TK[home_N[n],k1]*(V_KK[k1,k]+ V_KK[k,k1])
    #                     tau_tk += Z_NMKK[n,1,k1,k] + Z_NMKK[n,2,k,k1]
    #                 end
    #             elseif t == home_N[n]
    #                 for k1 in 1:model.K 
    #                     phi_tk += U_TK[away_N[n],k1]*(V_KK[k,k1] + V_KK[k1,k])
    #                     tau_tk += Z_NMKK[n,1,k,k1] + Z_NMKK[n,2,k1,k]
    #                 end
    #             end
    #         end
    #         post_shape = model.a + tau_tk
    #         post_rate = model.b + model.D*log(1/(model.p))*phi_tk
    #         U_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
    #     end
    # end

    # state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>I_NM)
    return data, state
end




