include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights_STARshared <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    a::Float64
    c::Float64
    d::Float64
    g::Function
    g_inv::Function
end

function STARlogpmf(model::flights_STARshared, x, mu, sigma2)
    std = sqrt(sigma2)

    if x == 0
        # logcdf for zero
        z = (model.g_inv(0) - mu) / std
        return logcdf(Normal(0,1), z)
    else
        # upper and lower
        z1 = (model.g(x) - mu) / std
        z0 = (model.g(x - 1) - mu) / std
        
        # use logcdf for both, then logdiffexp trick
        # log(cdf(z1) - cdf(z0)) = logcdf(z1) + log1mexp(logcdf(z0) - logcdf(z1))
        lc1 = logcdf(Normal(0,1), z1)
        lc0 = logcdf(Normal(0,1), z0)
        
        # ensure stability
        if lc1 < lc0
            # swap to avoid negative inside log
            return lc0 + log1mexp(lc1 - lc0)
        else
            return lc1 + log1mexp(lc0 - lc1)
        end
    end
end

function evalulateLogLikelihood(model::flights_STARshared, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    route = info["I_N3"][row,3]
    mu = state["U_R"][route]
    sigma2 = state["sigma2"]
    x =  STARlogpmf(model,Y,mu,sigma2)
    return x
end



function sample_likelihood(model::flights_STARshared, μ, σ, n::Int=1)
    f(z) = z < 0 ? 0 : ceil(model.g_inv(z))

    Z = rand(Normal(μ, σ), n)

    return n == 1 ? f(Z[1]) : f.(Z)
end

function sample_prior(model::flights_STARshared, info=nothing, constantint=nothing)
    U_R = rand(Normal(0, 1), model.R)
    sigma2 = rand(InverseGamma(model.c, model.d))

    state = Dict("U_R" => U_R,"beta"=>beta, "sigma2"=>sigma2,
                "I_N3"=>info["I_N3"], "routes_R4"=>info["routes_R4"],
                "airports_T4"=>info["airports_T4"]
                )
    return state
end

function forward_sample(model::flights_STARshared; state=nothing, info=nothing)
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

function backward_sample(model::flights_STARshared, data, state, mask=nothing; skipupdate=nothing)
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
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), -Inf, model.g(0)))
        else
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), model.g(Y_NM[n,1] - 1), model.g(Y_NM[n,1])))
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
