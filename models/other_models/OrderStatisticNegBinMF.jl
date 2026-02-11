include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct OrderStatisticNegBinMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
end

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end


function evalulateLogLikelihood(model::OrderStatisticNegBinMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    p = state["p"]
    if model.D == 1
        return logpdf(NegativeBinomial(mu, p), Y)
    else
        return logpdf(OrderStatistic(NegativeBinomial(mu, p), model.D, model.j), Y)
    end
end

function sample_prior(model::OrderStatisticNegBinMF)
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    p = rand(Beta(model.alpha,model.beta))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p"=>p)
    return state
end

function forward_sample(model::OrderStatisticNegBinMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    p = state["p"]
    Y_NM = zeros(Int, model.N, model.M)
   # p_NM = fill(p,model.N, model.M)
    for n in 1:model.N
        for m in 1:model.M
            Y_NM[n,m] = rand(OrderStatistic(NegativeBinomial(Mu_NM[n,m], p), model.D, model.j))
        end
    end

    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function griddy_gibbs(model::OrderStatisticNegBinMF, U_NK, Z_MK, Z_NM, plist=.01:.01:.99)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    rlist = zeros(length(plist),model.K, model.M)
    logprobs = zeros(length(plist))
    for (i,p) in enumerate(plist)
        #for each p, sample an r from its complete conditional
        post_rate = model.d + model.D*log(1/p)
        @views for k in 1:model.K
            @views for m in 1:model.M
                post_shape = model.c + Z_MK[m,k]
                rlist[i,k,m] = rand(Gamma(post_shape, 1/post_rate))
            end
        end
        mu_NM = U_NK * rlist[i,:,:]
        #calculate logprob for griddy gibbs (r,p) pair
        logprobs[i] = logpdf(Beta(model.alpha,model.beta),p) + sum(logpdf.(NegativeBinomial.(model.D*mu_NM, p), Z_NM))
    end

    c = argmax(rand(Gumbel(0,1),length(logprobs)) .+ logprobs)
    return (copy(plist[c]), copy(rlist[c,:,:]))
end

function backward_sample(model::OrderStatisticNegBinMF, data, state, mask=nothing; griddy=false)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    p = copy(state["p"])
    
   # if !isnothing(mask)
    Mu_NM = U_NK * V_KM
   #end
    #Loop over the non-zeros in Y_NM and allocate
    nt = Threads.nthreads()

    if griddy
            Z_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
        Z_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
        z1_thr = [0 for _ in 1:nt]
        P_K_thr = [zeros(Float64, model.K) for _ in 1:nt]
        Z_NM = zeros(model.N, model.M)
        @views @threads for idx in 1:(model.N * model.M)
            tid = Threads.threadid()
            n = div(idx - 1, model.M) + 1
            m = mod(idx - 1, model.M) + 1  
            y = Y_NM[n,m]
            if !isnothing(mask)
                if mask[n,m] == 1
                    y = rand(OrderStatistic(NegativeBinomial(Mu_NM[n,m],p), model.D, model.j))
                end
            end
            @assert y >= 0
            z1 = sampleSumGivenOrderStatistic(y, model.D, model.j, NegativeBinomial(Mu_NM[n,m],p))
            Z_NM[n,m] = z1
            z1_thr[tid] += z1
            if z1 > 0
                P_K = P_K_thr[tid]
                @inbounds for k in 1:model.K
                    P_K[k] = U_NK[n, k] * V_KM[k, m]
                end
                #sample CRT
                z = sampleCRT(z1, model.D*Mu_NM[n,m])
                
                #now Z is a (certain kind of) Poisson so we can thin it
                z_k = rand(Multinomial(z, P_K / sum(P_K)))
                @inbounds for k in 1:model.K
                    Z_NK_thr[tid][n, k] += z_k[k]
                    Z_MK_thr[tid][m, k] += z_k[k]
                end
            end
        end
        Z_NK  = sum(Z_NK_thr)  
        Z_MK  = sum(Z_MK_thr)  
        #z1sum = sum(z1_thr)

        A_K = fill(model.a, model.N)
        @views for k in 1:model.K
            U_NK[:, k] = rand(Dirichlet(A_K .+ Z_NK[:,k]))
        end
        if p <= .025
            plist = .001:.001:(p+.025)
        elseif p >= .975
            plist = (p-.025):.001:.999
        else
            plist = (p-.025):.001:(p+.025)
        end
        (p, V_KM) = griddy_gibbs(model, U_NK, Z_MK, Z_NM, plist)
    else
       
        Z_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
        Z_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
        z1_thr = [0 for _ in 1:nt]
        P_K_thr = [zeros(Float64, model.K) for _ in 1:nt]

        @views for idx in 1:(model.N * model.M)
           
            tid = Threads.threadid()
            n = div(idx - 1, model.M) + 1
            m = mod(idx - 1, model.M) + 1  
            y = Y_NM[n,m]
            if !isnothing(mask)
                if mask[n,m] == 1
                    y = rand(OrderStatistic(NegativeBinomial(Mu_NM[n,m], p), model.D, model.j))
                end
            end
            z1 = sampleSumGivenOrderStatistic(y, model.D, model.j, NegativeBinomial(Mu_NM[n,m],p))
            z1_thr[tid] += z1
            if z1 > 0
                P_K = P_K_thr[tid]
                @inbounds for k in 1:model.K
                    P_K[k] = U_NK[n, k] * V_KM[k, m]
                end
                #sample CRT
                z = sampleCRT(z1, model.D*Mu_NM[n,m])
                #now Z is a (certain kind of) Poisson so we can thin it
                z_k = rand(Multinomial(z, P_K / sum(P_K)))
                @inbounds for k in 1:model.K
                    Z_NK_thr[tid][n, k] += z_k[k]
                    Z_MK_thr[tid][m, k] += z_k[k]
                end
            end
        end
        # println(z1_thr)

        Z_NK  = sum(Z_NK_thr)  
        Z_MK  = sum(Z_MK_thr)  
        z1sum = sum(z1_thr)
        # println(z1sum)

    #     # mu = sum(U_NK * V_KM)
    #     # p = rand(Beta(model.alpha + model.D*mu, model.beta + z1sum))


        A_K = fill(model.a, model.N)
        @views for k in 1:model.K
            U_NK[:, k] = rand(Dirichlet(A_K .+ Z_NK[:,k]))
        end

        mu_sum = sum(U_NK * V_KM)
        # println(z1sum)
        # println(model.D*mu_sum)
        @assert model.D == 3
        p = rand(Beta(model.alpha + model.D*mu_sum, model.beta + z1sum))

        post_rate = model.d + model.D*log(1/p)
        @views for k in 1:model.K
            @views for m in 1:model.M
                post_shape = model.c + Z_MK[m,k]
                V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
            end
        end
        #p = rand(Beta(model.alpha + z1sum, model.beta + model.D*mu_sum ))
        
    #     # p = .675
    end


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "p"=>p)
    return data, state
end

