include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights_STAR2 <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    R::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    g::Function
    g_inv::Function
end

function evalulateLogLikelihood(model::flights_STAR2, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    home = info["I_N3"][row,1]
    away = info["I_N3"][row,2]
    route = info["I_N3"][row,3]
    dist = info["dist_N"][row,1]

    A_T = state["A_T"]
    B_T = state["B_T"]

    #mu = A_T[home] + B_T[away] + dist*state["U_R"][route]
    #mu = A_T[home] + B_T[away] + state["U_R"][route]
    mu = state["U_R"][route]

    if Y == 0
        return log(cdf(Normal(0,1), (model.g(0.5) - mu)))
    else 
        return log(cdf(Normal(0,1), (model.g(Y + .5) - mu)) - cdf(Normal(0,1), (model.g(Y - .5) - mu)))
    end
end



function sample_likelihood(model::flights_STAR2, μ, σ, n::Int=1)
    f(z) = z < 0.5 ? 0 : Int(round(model.g_inv(z)))

    Z = rand(Normal(μ, σ), n)

    return n == 1 ? f(Z[1]) : f.(Z)
end

function sample_prior(model::flights_STAR2, info=nothing, constantint=nothing)
    #beta = rand(Gamma(model.c, 1/model.d))
    #sigma2 = rand(InverseGamma(model.a, model.b))


    U_R = rand(Normal(0, 1), model.R)
    A_T = rand(Normal(0, 1), model.T)
    B_T = rand(Normal(0, 1), model.T)

    state = Dict("U_R" => U_R, "A_T" => A_T, "B_T" => B_T, #"beta"=>beta, 
    #"sigma2"=>sigma2,
                "I_N3"=>info["I_N3"], "dist_N" => info["dist_N"], "routes_R4"=>info["routes_R4"],
                "airports_T4"=>info["airports_T4"]
                )
    return state
end

function forward_sample(model::flights_STAR2; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    #Z_TT = state["Z_TT"]
    U_R = state["U_R"]

    A_T = state["A_T"]
    B_T = state["B_T"]

    I_N3 = state["I_N3"]
    dist_N = state["dist_N"]
    
    home_N = state["I_N3"][:,1]
    away_N = state["I_N3"][:,2]
    routes_N = state["I_N3"][:,3]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        route = routes_N[n]
        #mu = A_T[home] + B_T[away] + dist_N[n,1]*U_R[route]
        #mu = A_T[home] + B_T[away] + U_R[route]
        mu = U_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,1)
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function backward_sample(model::flights_STAR2, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_R = copy(state["U_R"])
    #beta = copy(state["beta"])
    #sigma2 = copy(state["sigma2"])
    I_N3 = copy(state["I_N3"])
    A_T = copy(state["A_T"])
    B_T = copy(state["B_T"])
    dist_N = copy(state["dist_N"])
    info = Dict("I_N3"=>I_N3,"dist_N"=>dist_N)

    home_N = I_N3[:,1]
    away_N = I_N3[:,2]
    routes_N = I_N3[:,3]
    @assert model.M == 1
    routes_R4 = state["routes_R4"]
    airports_T4 = state["airports_T4"]

    Z_NM = zeros(Float64, model.N,1)



    @views @threads for n in 1:model.N   
        home = home_N[n]
        away = away_N[n]
        
        #mu = A_T[home]+B_T[away]+dist_N[n,1]*U_R[routes_N[n]]
        #mu = A_T[home]+B_T[away]+dist_N[n,1]*U_R[routes_N[n]]
        mu = U_R[routes_N[n]]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = sample_likelihood(model,mu,1)
        end
        if Y_NM[n,1] == 0
            Z_NM[n,1] = rand(Truncated(Normal(mu, 1), -Inf, model.g(.5)))
        else
            Z_NM[n,1] = rand(Truncated(Normal(mu, 1), model.g(Y_NM[n,1] - .5), model.g(Y_NM[n,1] + .5)))
        end
    end

    @views for r in 1:model.R
        #access pre-calculated route info
        home = routes_R4[r,1]
        away = routes_R4[r,2]
        indices = routes_R4[r,3]
        distance = routes_R4[r,4]

        n = length(indices)
        #s = sum(Z_NM[indices,1]) - n*A_T[home] - n*B_T[away]
        s = sum(Z_NM[indices,1]) #- n*A_T[home] - n*B_T[away]
        # =sigma2_R[r]


        #V = 1 / (1/ sigma2 + (distance^2 * n) )
        # V = 1 / (1 + (distance^2 * n) )
        # m = V * (distance * s)
         V = 1 / (1 + (n) )
        m = V * (s)
        U_R[r] = rand(Normal(m,sqrt(V)))
    end

    # @views for t in 1:model.T
    #     #access pre-calculated route info
    #     indices = airports_T4[t,1] #get flights by origin
    #     routes = airports_T4[t,2]
    #     destinations = routes_R4[routes,2]
    #     distances = routes_R4[routes,4]
    #     n = length(indices)
    #     s = 0.0
    #     for i in 1:length(indices)
    #         f = indices[i]
    #         #s += (Z_NM[f,1] - B_T[destinations[i]] - U_R[routes[i]] * distances[i])
    #         s += (Z_NM[f,1] - B_T[destinations[i]] - U_R[routes[i]])
    #     end
       

    #     V = 1 / (1 + n)
    #     m = V * s
    #     A_T[t] = rand(Normal(m,sqrt(V)))
    # end

    #A_T .-= mean(A_T)


    # @views for t in 1:model.T
    #     #access pre-calculated route info
    #     indices = airports_T4[t,3] #get flights by destination
    #     routes = airports_T4[t,4]
    #     origins = routes_R4[routes,1]
    #     distances = routes_R4[routes,4]
    #     n = length(indices)
    #     s = 0.0
    #     for i in 1:length(indices)
    #         f = indices[i]
    #         #s += (Z_NM[f,1] - A_T[origins[i]] - U_R[routes[i]] * distances[i])
    #         s += (Z_NM[f,1] - A_T[origins[i]] - U_R[routes[i]])
    #     end

    #     V = 1 / (1 + n)
    #     m = V * s
    #     B_T[t] = rand(Normal(m,sqrt(V)))
    # end

    #B_T .-= mean(B_T)

    # #update sigma2
    # @views for r in 1:model.R
    #     sigma2_R[r] = rand(InverseGamma(model.a + 1/2, model.b + (U_R[r]^2)/2))
    # end
    #sigma2 = rand(InverseGamma(model.a + model.R/2, model.b + sum(U_R.^2)/2))

    # #update gamma hyperprior
    #beta = rand(Gamma(model.c + model.R*model.a, 1/(model.d + sum(1 ./ sigma2_R))))


    state = Dict("U_R" => U_R, "A_T"=>A_T, "B_T"=>B_T,#"sigma2"=>sigma2,#"beta"=>beta,
     "I_N3"=>I_N3, "dist_N" => dist_N, "routes_R4"=>routes_R4, "airports_T4"=>airports_T4)
    return data, state
end