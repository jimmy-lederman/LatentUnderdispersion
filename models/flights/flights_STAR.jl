include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights_STAR <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    a::Float64
    b::Float64
    tau2::Float64
    g::Function
    g_inv::Function
end

function STARlogpmf(model::flights_STAR, x, mu, sigma2)
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

function evalulateLogLikelihood(model::flights_STAR, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    route = data["routes_N"][row]
    mu = state["U_R"][route]
    sigma2 = state["sigma2_R"][route]
    x =  STARlogpmf(model,Y,mu,sigma2)
    return x
end



function sample_likelihood(model::flights_STAR, μ, σ, n::Int=1)
    f(z) = z < 0 ? 0 : ceil(model.g_inv(z))

    Z = rand(Normal(μ, σ), n)

    return n == 1 ? f(Z[1]) : f.(Z)
end

function sample_prior(model::flights_STAR, info=nothing, constantint=nothing)
    U_R = rand(Normal(0, sqrt(model.tau2)), model.R)
    sigma2_R = rand(InverseGamma(model.a, model.b), model.R)

    state = Dict("U_R" => U_R,"sigma2_R"=>sigma2_R
                )
    return state
end

function forward_sample(model::flights_STAR; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    U_R = state["U_R"]
    I_N3 = state["I_N3"]
    sigma2_R =state["sigma2_R"]
    routes_N = state["I_N3"][:,3]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        route = routes_N[n]
        mu = U_R[route]
        sigma2 = sigma2_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,sqrt(sigma2))
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function backward_sample(model::flights_STAR, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = data["Y_NM"]
    routes_R2 = data["routes_R2"]
    routes_N = data["routes_N"]
    U_R = copy(state["U_R"])
    sigma2_R = copy(state["sigma2_R"])
    @assert model.M == 1
    
    Z_NM = zeros(Float64, model.N,1)


    @views @threads for n in 1:model.N   
        r = routes_N[n]
        mu = U_R[r]
        sigma2 = sigma2_R[r]
        if !isnothing(mask)
            if mask[n,1] == 1
                z = rand(Normal(mu, sqrt(sigma2)))
                if z < 0
                    Y_NM[n,1] = 0
                else
                    Y_NM[n,1] = ceil(model.g_inv(z))
                end
            end
        end
        if Y_NM[n,1] == 0
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), -Inf, model.g(0)))
        else
            Z_NM[n,1] = rand(Truncated(Normal(mu, sqrt(sigma2)), model.g(Y_NM[n,1] - 1), model.g(Y_NM[n,1])))
        end
    end

    @views for r in 1:model.R
        #access pre-calculated route info
        indices = routes_R2[r,1]
        n = routes_R2[r,2]
        sigma2 =  sigma2_R[r]
        z_N = Z_NM[indices,1]
        s = sum(z_N)

        V = 1 / (n / sigma2 + 1 / model.tau2)
        m = V * (s / sigma2)
        U_R[r] = rand(Normal(m, sqrt(V)))
        
        s2 = sum((z_N .- U_R[r]).^2)
        sigma2_R[r] = rand(InverseGamma(model.a + n/2, model.b+ s2/2))
    end

    state = Dict("U_R" => U_R, "sigma2_R"=>sigma2_R)
    return data, state
end
