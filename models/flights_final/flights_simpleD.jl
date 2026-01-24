include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flightsDsimple <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    R::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
end

function evalulateLogLikelihood(model::flightsDsimple, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    route = info["I_N3"][row,3]
    mu = state["U_R"][route]
    D = state["D_R"][route]
    
    if D == 1
        return logpdf(Poisson(mu), Y)
    else
        return logpmfOrderStatPoisson(Y,mu,D,div(D,2)+1)
    end
end



function sample_likelihood(model::flightsDsimple, mu,D,n=1)
    if D == 1
        if n == 1
            return rand(Poisson(mu))
        else
            return rand(Poisson(mu),n)
        end
    else
        if n == 1
            j = div(D,2) + 1
            return rand(OrderStatistic(Poisson(mu), D, j))
        else
            j = div(D,2) + 1
            return rand(OrderStatistic(Poisson(mu), D, j),n)
        end
    end
end

function sample_prior(model::flightsDsimple, info=nothing, constantint=nothing)
    p = nothing
    U_R = rand(Gamma(model.a, 1/model.b), model.R)
    p = rand(Beta(model.alpha,model.beta))
    @assert mod(model.Dmax, 2) == 1
    D_R = 2 * rand(Binomial((model.Dmax - 1)/2, p), model.R) .+ 1

    state = Dict("U_R" => U_R, "p"=>p, "D_R"=>D_R,
                "I_N3"=>info["I_N3"], 
                "routes_R4"=>info["routes_R4"],
                )
    return state
end

function forward_sample(model::flightsDsimple; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U_R = state["U_R"]

    I_N3 = state["I_N3"]
    D_R =state["D_R"]
    routes_N = state["I_N3"][:,3]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        route = routes_N[n]
        mu = U_R[route]
        D = D_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,D)
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function backward_sample(model::flightsDsimple, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_R = copy(state["U_R"])
    D_R = copy(state["D_R"])
    p = copy(state["p"])

    I_N3 = copy(state["I_N3"])
    routes_R4 = state["routes_R4"]

    routes_N = I_N3[:,3]
    @assert model.M == 1

    #3: the distance*cluster mean

    post_shape_R = fill(model.a, model.R)
    post_rate_R = fill(model.b, model.R)


    Z1_NM = zeros(Int, model.N,1)

    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    @views @threads for n in 1:model.N   
        mu = U_R[routes_N[n]]
        D = D_R[routes_N[n]]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = sample_likelihood(model,mu,D)
        end
        #if Y_NM[n, 1] > 0
        Z1_NM[n,1] = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, div(D,2)+1, Poisson(mu))
        #end
    end


    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    @views for r in 1:R
        indices = routes_R4[r,3]
        numflights = length(indices)
        post_shape_R[r] += sum(Z1_NM[indices,1])
        post_rate_R[r] += D_R[r]*numflights
    end

    U_R = rand.(Gamma.(post_shape_R, 1 ./post_rate_R))


    if isnothing(skipupdate) || !("D_R" in skipupdate)
        @views for r in 1:model.R
            home = routes_R4[r,1]
            away = routes_R4[r,2]
            indices = routes_R4[r,3]
            #dist = routes_R4[r,4]
            logprob = 0
            mu = U_R[r]
            
            logprobs = [logbinomial(Int((model.Dmax-1)/2),Int((d-1)/2)) + (d-1)*log(p)/2 + (model.Dmax - d)*log(1-p)/2 for d in 1:2:model.Dmax]
            for d in 1:2:model.Dmax
                @threads for f in 1:length(indices)
                    Y = Y_NM[indices[f], 1]
                    try
                        logprobs[div(d,2)+1] += logpmfOrderStatPoisson(Y,mu,d,div(d,2)+1)
                    catch ex
                        println(Y," ", mu, " ", d," ", div(d,2)+1)
                        @assert 1 == 2
                    end
                end
            end
            D_R[r] = 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
        end
        #update p
        p = rand(Beta(model.alpha + (sum(D_R)- model.R)/2, model.beta + (model.Dmax*R - sum(D_R))/2))
    end

    state = Dict("U_R" => U_R,
                "p"=>p,
                "D_R"=>D_R,
                "I_N3"=>I_N3, 
                "routes_R4"=>routes_R4)
    return data, state
end