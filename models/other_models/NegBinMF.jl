include("../../helper/MatrixMF.jl")
include("../../helper/NegBinPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

struct NegBinMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
end

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end

function evalulateLogLikelihood(model::NegBinMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    p = state["p"]
    return logpdf(NegativeBinomial(mu, p), Y)
end

function sample_prior(model::NegBinMF)
    dhyper = sample_rate_prior()   # hierarchical Gamma rate for V
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1/dhyper), model.K, model.M)
    p = rand(Beta(model.alpha,model.beta))
    dhyper = sample_rate_hyper(V_KM, model.c)   # d | V, conjugate
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "d" => dhyper, "p"=>p)
    return state
end

function forward_sample(model::NegBinMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    p = state["p"]
    Y_NM = zeros(Int, model.N, model.M)
   # p_NM = fill(p,model.N, model.M)
    for n in 1:model.N
        for m in 1:model.M
            Y_NM[n,m] = rand(NegativeBinomial(Mu_NM[n,m], p))
        end
    end

    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function griddy_gibbs(model::NegBinMF, U_NK, Z_MK, Y_NM, plist=.01:.01:.99)#plist=[.49,.5,.51,.52,.53,.55,.6]))
    rlist = zeros(length(plist),model.K, model.M)
    logprobs = zeros(length(plist))
    for (i,p) in enumerate(plist)
        #for each p, sample an r from its complete conditional
       
        @views for k in 1:model.K
            post_rate = dhyper + log(1/p)
            @views for m in 1:model.M
                post_shape = model.c + Z_MK[m,k]
                rlist[i,k,m] = rand(Gamma(post_shape, 1/post_rate))
            end
        end
        # accumulate the grid log-likelihood without materializing mu_NM, the array of
        # NegativeBinomial objects, or the array of logpdfs (4 N*M temporaries per grid
        # point x length(plist) points). Same quantity; only the summation order differs.
        acc = 0.0
        @inbounds for m in 1:model.M
            for n in 1:model.N
                mu = 0.0
                for k in 1:model.K
                    mu += U_NK[n,k] * rlist[i,k,m]
                end
                acc += logpdf(NegativeBinomial(mu, p), Y_NM[n,m])
            end
        end
        logprobs[i] = logpdf(Beta(model.alpha,model.beta),p) + acc
    end

    c = argmax(rand(Gumbel(0,1),length(logprobs)) .+ logprobs)
    return (plist[c], rlist[c,:,:])
end

function backward_sample(model::NegBinMF, data, state, mask=nothing; skipupdate = nothing, griddy=false)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    dhyper = state["d"]
    p = state["p"]
    Mu_NM = U_NK * V_KM
    nt = Threads.nthreads()
    Z_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Z_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K) for _ in 1:nt]
    zk_thr = [zeros(Int, model.K) for _ in 1:nt]


    @views @threads for idx in 1:(model.N * model.M)
        tid = Threads.threadid()
        n = div(idx - 1, model.M) + 1
        m = mod(idx - 1, model.M) + 1  
        if !isnothing(mask)
            if mask[n,m] == 1
                Y_NM[n,m] = rand(NegativeBinomial(Mu_NM[n,m],p))
            end
        end
        if Y_NM[n, m] > 0
            P_K = P_K_thr[tid]
            @inbounds for k in 1:model.K
                P_K[k] = U_NK[n, k] * V_KM[k, m]
            end
            #sample CRT
            z = sampleCRT(Y_NM[n,m],Mu_NM[n,m])
            #now Z is a (certain kind of) Poisson so we can thin it
            z_k = thin_multinomial!(zk_thr[tid], z, P_K, model.K)
            @inbounds for k in 1:model.K
                Z_NK_thr[tid][n, k] += z_k[k]
                Z_MK_thr[tid][m, k] += z_k[k]
            end
        end
    end
    Z_NK  = sum(Z_NK_thr)  
    Z_MK  = sum(Z_MK_thr)  

    A_K = fill(model.a, model.N)
    @views for k in 1:model.K
        U_NK[:, k] = rand(Dirichlet(A_K .+ Z_NK[:,k]))
    end

    if griddy
        (p, V_KM) = griddy_gibbs(model, U_NK, Z_MK, Y_NM)
    else
        @views for k in 1:model.K
            post_rate = dhyper + log(1/p)
            @views for m in 1:model.M
                post_shape = model.c + Z_MK[m,k]
                V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))
            end
        end
        
        # conjugate dispersion update. Y ~ NB(r, p) with r = (UV) gives a
        # likelihood p^{sum r} (1-p)^{sum y}, so with p ~ Beta(alpha, beta):
        #     p | - ~ Beta(alpha + sum(UV), beta + sum(Y))
        # Y_NM here has the held-out entries imputed, which is what the
        # augmented conditional requires. This ran only inside the griddy
        # branch before, so p was frozen after iteration 250.
        p = rand(Beta(model.alpha + sum(U_NK * V_KM), model.beta + sum(Y_NM)))
    end


    dhyper = sample_rate_hyper(V_KM, model.c)   # d | V, conjugate


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "d" => dhyper, "p"=>p)
    return data, state
end
