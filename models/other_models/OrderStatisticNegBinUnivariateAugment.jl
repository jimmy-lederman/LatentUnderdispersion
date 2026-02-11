include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/NegBinPMF.jl")
#include("../../helper/OrderStatsSampling_old.jl")
using Distributions
using Base.Threads

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end

struct OrderStatisticNegBinUnivariate <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
end

function evalulateLogLikelihood(model::OrderStatisticNegBinUnivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    p = state["p"]
    return logpmfOrderStatNegBin(Y, mu, p, model.D, model.j)
end

function sample_prior(model::OrderStatisticNegBinUnivariate, info=nothing,constantinit=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    p = rand(Beta(model.alpha, model.beta))
    state = Dict("mu" => mu, "p" => p)
    return state
end

function forward_sample(model::OrderStatisticNegBinUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    p = state["p"]

    # Y_ND = rand(NegativeBinomial(mu, 1-p), model.N, model.D)
    # Z1_N = sum(Y_ND,dims=2)
    #Y_NM = reshape(sort(Y_ND,dims=2)[:,model.j],model.N,1)

    Y_NM = rand(OrderStatistic(NegativeBinomial(mu, p), model.D, model.j), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    state = Dict("mu"=>mu,"p"=>p)
    return data, state 
end


function backward_sample(model::OrderStatisticNegBinUnivariate, data, state, mask=nothing;annealStrat=nothing,anneal=nothing,)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    p = copy(state["p"])
    Z1_ND = zeros(model.N, model.D)
    #Z1_N = copy(state["Z1_N"])
    # Z2_N = zeros(model.N)
    # println(mu)
    # @assert model.D == 1
    # println(mean(Y_NM))
    # @assert 1 == 2
    
    nt = Threads.nthreads()
    z1sum_nt = [0 for _ in 1:nt]
    z2sum_nt = [0 for _ in 1:nt]
    @views @threads for n in 1:model.N
        tid = Threads.threadid()
        #x = sampleAllGivenOrderStatistic(Y_NM[n,1], model.D, model.j, NegativeBinomial(mu, p))
        z1 = sampleSumGivenOrderStatistic(Y_NM[n,1], model.D, model.j, NegativeBinomial(mu, p))
        z1sum_nt[tid] += z1
        z2 = sampleCRT(z1, model.D*mu)
        z2sum_nt[tid] += z2
    end
    z1sum = sum(z1sum_nt)
    z2sum = sum(z2sum_nt)
    
    if !isnothing(annealStrat)
        post_shape = model.a + z2sum
        post_rate = model.b + log(1/p)*model.N*model.D
        mu = rand(Gamma(post_shape/anneal, anneal/post_rate))
    else 
        post_shape = model.a + z2sum
        post_rate = model.b + log(1/p)*model.N*model.D
        mu = rand(Gamma(post_shape, 1/post_rate))
    end

    # Y = vec(z1sum)   # D=j=1 case
    # mu = update_r_slice(
    #         mu,
    #         vec(Z1_ND),
    #         p,
    #         a = model.a,
    #         b = model.b
    #     )


    #p = rand(Beta(model.beta + sum(Y_NM), model.alpha + mu*model.N*model.D))
    p = rand(Beta(model.alpha + mu*model.N*model.D, model.beta + z1sum))

    # nt = Threads.nthreads()
    # z1sum_nt = [0 for _ in 1:nt]
    # z2sum_nt = [0 for _ in 1:nt]
    # @views @threads for n in 1:model.N
    #     tid = Threads.threadid()
    #     z1 = sampleSumGivenOrderStatistic(Y_NM[n,1], model.D, model.j, NegativeBinomial(mu, p))
    #     z1sum_nt[tid] += z1
    #     z2 = sampleCRT(z1, model.D*mu)
    #     z2sum_nt[tid] += z2
    # end
    # z1sum = sum(z1sum_nt)
    # z2sum = sum(z2sum_nt)

    # u = rand(Gamma((z2sum + 1), 1/((model.D*model.N*mu + 1))))
    # mu = rand(Gamma(z2sum + model.a, 1/(model.D*model.N*u + model.b)))

    # t = rand(Gamma((z2sum + 1), 1/((model.N*mu*model.D + 1))))
    # p = 1 - exp(-t)

    #p = rand(Beta(model.beta + z1sum, model.alpha + mu*model.N*model.D))



    #p = .25

    # if !isnothing(annealStrat)
    #     
    # else
        
    # end
    


    
    # if !isnothing(annealStrat)
    #     t = rand(Gamma((z + 1), 1/((model.N*mu*model.D + 1))))
    #     p = 1 - exp(-t)
    # else
    
    #end
    
    #p = .5
    # println(p)
    # println(model.N*mu*model.D)
    # println(sum(Z1_N))
    #p = rand(Beta(model.beta + model.N*mu*model.D, model.alpha + sum(Z1_N)))
    # println(p)

    state = Dict("mu" => mu, "p" => p)
    return data, state
end

function logpost_r(r, Y, p, a, b)
    r <= 0 && return -Inf

    ll = 0.0
    @inbounds for y in Y
        ll += loggamma(y + r) - loggamma(r) + r*log(p)
    end

    lp = (a - 1)*log(r) - b*r
    return ll + lp
end

using Random, SpecialFunctions

function update_r_slice(
    r_cur::Float64,
    Y,
    p::Float64;
    a::Float64,
    b::Float64,
    w::Float64 = 1.0,     # initial bracket width
    m::Int = 100          # max step-out steps
)
    logf = r -> logpost_r(r, Y, p, a, b)

    logy = logf(r_cur) - randexp()

    # --- step out ---
    u = rand()
    L = max(r_cur - w*u, 1e-12)
    R = L + w

    j = rand(0:m)
    k = (m - j)

    while j > 0 && logf(L) > logy
        L = max(L - w, 1e-12)
        j -= 1
    end

    while k > 0 && logf(R) > logy
        R += w
        k -= 1
    end

    # --- shrink ---
    while true
        r_new = rand()*(R - L) + L
        if logf(r_new) >= logy
            return r_new
        elseif r_new < r_cur
            L = r_new
        else
            R = r_new
        end
    end
end

