include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights_STARsimple <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    R::Int64
    a::Float64
    c::Float64
    d::Float64
    g::Function
    g_inv::Function
end

function evalulateLogLikelihood(model::flights_STARsimple, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    route = info["I_N3"][row,3]
    mu = state["U_R"][route]
    sigma2 = state["sigma2"]
    # if row == 41068
    #     println(mu)
    #     println(Y)
    #     println(sigma2)
    #     println(cdf(Normal(0,1), (model.g(Y + .5) - mu)/sqrt(sigma2)))
    #     println(cdf(Normal(0,1), (model.g(Y - .5) - mu)/sqrt(sigma2)))
    #     @assert 1 == 2
    # end

    if Y == 0
        return log(cdf(Normal(0,1), (model.g(0.5) - mu)/sqrt(sigma2)))
    else 
        a = (model.g(Y - 0.5) - mu) / sqrt(sigma2)
        b = (model.g(Y + 0.5) - mu) / sqrt(sigma2)

        logb = logcdf(Normal(), b)
        loga = logcdf(Normal(), a)
        return logsubexp(logb, loga)
        #return log(cdf(Normal(0,1), (model.g(Y + .5) - mu)/sqrt(sigma2)) - cdf(Normal(0,1), (model.g(Y - .5) - mu)/sqrt(sigma2)))
    end
end



function sample_likelihood(model::flights_STARsimple, μ, σ, n::Int=1)
    f(z) = z < 0.5 ? 0 : Int(round(model.g_inv(z)))

    Z = rand(Normal(μ, σ), n)

    return n == 1 ? f(Z[1]) : f.(Z)
end

function sample_prior(model::flights_STARsimple, info=nothing, constantint=nothing)
    U_R = rand(Normal(0, 1), model.R)
    sigma2 = rand(InverseGamma(model.c, model.d))

    state = Dict("U_R" => U_R,"beta"=>beta, "sigma2"=>sigma2,
                "I_N3"=>info["I_N3"], "routes_R4"=>info["routes_R4"],
                "airports_T4"=>info["airports_T4"]
                )
    return state
end

function forward_sample(model::flights_STARsimple; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    U_R = state["U_R"]
    I_N3 = state["I_N3"]
    sigma2 =state["sigma2"]
    routes_N = state["I_N3"][:,3]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        route = routes_N[n]
        mu = U_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,sqrt(sigma2))
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function backward_sample(model::flights_STARsimple, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_R = copy(state["U_R"])
    sigma2 = copy(state["sigma2"])
    I_N3 = copy(state["I_N3"])
    routes_N = I_N3[:,3]
    @assert model.M == 1
    routes_R4 = state["routes_R4"]
    airports_T4 = state["airports_T4"]
    Z_NM = zeros(Float64, model.N,1)

    @views @threads for n in 1:model.N   
        mu = U_R[routes_N[n]]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = sample_likelihood(model,mu,sqrt(sigma2))
        end
        if Y_NM[n,1] == 0
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), -Inf, model.g(.5)))
        else
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), model.g(Y_NM[n,1] - .5), model.g(Y_NM[n,1] + .5)))
        end
    end

    @views for r in 1:model.R
        #access pre-calculated route info
        indices = routes_R4[r,3]

        n = length(indices)
        s = sum(Z_NM[indices,1])


        V = 1 / (1 + (n) / sigma2)
        m = V * (s / sigma2)
        U_R[r] = rand(Normal(m,sqrt(V)))
    end


    s = sum((Z_NM .- U_R[routes_N]).^2)
    sigma2 = rand(InverseGamma(model.c + model.N/2, model.d+ s/2))

    state = Dict("U_R" => U_R, "sigma2"=>sigma2,
     "I_N3"=>I_N3, "routes_R4"=>routes_R4, "airports_T4"=>airports_T4)
    return data, state
end
