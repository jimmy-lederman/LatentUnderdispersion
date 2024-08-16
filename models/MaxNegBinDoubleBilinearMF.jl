include("MatrixMF.jl")
include("PoissonMaxFunctions.jl")
using Distributions
using PyCall
using LogExpFunctions

struct MaxNegBinDoubleBilinear <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    D::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    p::Float64
    
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
        mu = state["U_TK"][home,:]' * state["V_KK"] * state["U_TK"][away,:]
    else
        mu = state["U_TK"][away,:]' * state["V_KK"] * state["U_TK"][home,:]
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
    U_TK = rand(Gamma(model.a, 1/model.b), model.T, model.K)
    V_KK = rand(Gamma(model.c, 1/model.d), model.K, model.K)
    @assert !isnothing(info)
    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>info["I_NM"])
    return state
end

function forward_sample(model::MaxNegBinDoubleBilinear; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U_TK = state["U_TK"]
    V_KK = state["V_KK"]
    
    Y_G2 = zeros(model.N, model.M)
    @assert M == 2
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]
    #Y_NMKK = zeros(model.N, model.M, model.K, model.K)
    Y_NM = zeros(Int, model.N,model.M)
    Mu_TT = U_TK * V_KK * U_TK'
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
    U_TK = copy(state["U_TK"])
    V_KK = copy(state["V_KK"])
    I_NM = copy(state["I_NM"])
    #Y_NMKK = copy(data["Y_NMKK"])
    Z1_NM = zeros(Int, model.N,model.M)
    Z2_NM = zeros(Int, model.N,model.M)
    Z_NMKK = zeros(Int, model.N, model.M, model.K, model.K)
    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 2
    
    P_KtimesK = zeros(model.K*model.K)
    MU_TT = U_TK * V_KK * U_TK'
    # Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        @assert home != away
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    if m == 1 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[home,away],model.p),model.D,model.D)) end
                    if m == 2 Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(MU_TT[away,home],model.p),model.D,model.D)) end
                end
            end
            if Y_NM[n, m] > 0
                #sample latent poissons
                if m == 1
                    Z1_NM[n,m] = sampleSumGivenMax(Y_NM[n, m], model.D, NegativeBinomial(MU_TT[home, away],model.p))
                    Z2_NM[n,m] = sampleCRT(Z1_NM[n,m],model.D*MU_TT[home, away])
                else
                    Z1_NM[n,m] = sampleSumGivenMax(Y_NM[n, m], model.D, NegativeBinomial(MU_TT[away, home],model.p))
                    Z2_NM[n,m] = sampleCRT(Z1_NM[n,m],model.D*MU_TT[away, home])
                end
                
                if m == 1
                    k = 1
                    for k1 in 1:K
                        for k2 in 1:K
                            P_KtimesK[k] = U_TK[home,k1]*V_KK[k1,k2]*U_TK[away,k2]
                            k += 1
                        end
                    end
                    P_KtimesK /= sum(P_KtimesK)
                    #sample vector of length K*K
                    vec_KtimesK = rand(Multinomial(Z2_NM[n, m], P_KtimesK))
                    #reshape into K by K matrix
                    Z_NMKK[n,m,:,:] = permutedims(reshape(vec_KtimesK, model.K, model.K))
                else
                    k = 1
                    for k1 in 1:K
                        for k2 in 1:K
                            P_KtimesK[k] = U_TK[away,k1]*V_KK[k1,k2]*U_TK[home,k2]
                            k += 1
                        end
                    end
                    P_KtimesK /= sum(P_KtimesK)
                    #sample vector of length K*K
                    vec_KtimesK = rand(Multinomial(Z2_NM[n, m], P_KtimesK))
                    #reshape into K1 by K2 matrix
                    Z_NMKK[n,m,:,:] = permutedims(reshape(vec_KtimesK, model.K, model.K))
                end
            end
        end
    end

    #update the model parameters
    #first update V_KK
    for k1 in 1:model.K
        for k2 in 1:model.K
            beep_k1k2 = 0
            boop_k1k2 = 0
            for n in 1:model.N
                @assert model.M == 2
                beep_k1k2 += Z_NMKK[n,1,k1,k2] + Z_NMKK[n,2,k1,k2]
                boop_k1k2 += U_TK[home_N[n],k1]*U_TK[away_N[n],k2] + U_TK[away_N[n],k1]*U_TK[home_N[n],k2]
            end
            post_shape = model.c + beep_k1k2
            post_rate = model.d + model.D*log(1/(model.p))*boop_k1k2
            V_KK[k1,k2] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    #now update U_TK
    for t in 1:model.T
        for k in 1:model.K
            #calcultate phi_tk and tau_tk
            phi_tk = 0
            tau_tk = 0
            for n in 1:model.N 
                if t == away_N[n]
                    for k1 in 1:model.K 
                        phi_tk += U_TK[home_N[n],k1]*(V_KK[k1,k]+ V_KK[k,k1])
                        tau_tk += Z_NMKK[n,1,k1,k] + Z_NMKK[n,2,k,k1]
                    end
                elseif t == home_N[n]
                    for k1 in 1:model.K 
                        phi_tk += U_TK[away_N[n],k1]*(V_KK[k,k1] + V_KK[k1,k])
                        tau_tk += Z_NMKK[n,1,k,k1] + Z_NMKK[n,2,k1,k]
                    end
                end
            end
            post_shape = model.a + tau_tk
            post_rate = model.b + model.D*log(1/(model.p))*phi_tk
            U_TK[t,k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    state = Dict("U_TK" => U_TK, "V_KK" => V_KK, "I_NM"=>I_NM)
    return data, state
end




