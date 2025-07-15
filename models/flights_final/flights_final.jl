include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights <: MatrixMF
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

function evalulateLogLikelihood(model::flights, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    home = info["I_N3"][row,1]
    away = info["I_N3"][row,2]
    route = info["I_N3"][row,3]
    dist = info["dist_N"][row,1]

    A_T = state["A_T"]
    B_T = state["B_T"]

    mu = A_T[home] + B_T[away] + dist*state["U_R"][route]
    D = state["D_R"][route]
    
    if D == 1
        return logpdf(Poisson(mu), Y)
    else
        return logpmfOrderStatPoisson(Y,mu,D,div(D,2)+1)
    end
end



function sample_likelihood(model::flights, mu,D,n=1)
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

function sample_prior(model::flights, info=nothing, constantint=nothing)
    p = nothing
    #Z_TT = rand(Categorical(model.K), model.T, model.T)
    U_R = rand(Gamma(model.a, 1/model.b), model.R)
    A_T = rand(Gamma(model.c, 1/model.d), model.T)
    B_T = rand(Gamma(model.c, 1/model.d), model.T)
    p = rand(Beta(model.alpha,model.beta))
    #@assert !isnothing(info)
    @assert mod(model.Dmax, 2) == 1
    D_R = 2 * rand(Binomial((model.Dmax - 1)/2, p), model.R) .+ 1


    state = Dict("U_R" => U_R, "A_T" => A_T, "B_T" => B_T, "D_R"=>D_R, "p"=>p,
                "I_N3"=>info["I_N3"], "dist_N" => info["dist_N"], "routes_R4"=>info["routes_R4"],
                )
    return state
end

function forward_sample(model::flights; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    #Z_TT = state["Z_TT"]
    U_R = state["U_R"]

    A_T = state["A_T"]
    B_T = state["B_T"]

    I_N3 = state["I_N3"]
    dist_N = state["dist_N"]

    D_R =state["D_R"]
    
    home_N = state["I_N3"][:,1]
    away_N = state["I_N3"][:,2]
    routes_N = state["I_N3"][:,3]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        route = routes_N[n]
        mu = A_T[home] + B_T[away] + dist_N[n,1]*U_R[route]
        D = D_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,D)
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

function backward_sample(model::flights, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_R = copy(state["U_R"])
    D_R = copy(state["D_R"])
    I_N3 = copy(state["I_N3"])
    A_T = copy(state["A_T"])
    B_T = copy(state["B_T"])
    p = copy(state["p"])
    dist_N = copy(state["dist_N"])
    info = Dict("I_N3"=>I_N3,"dist_N"=>dist_N)

    home_N = I_N3[:,1]
    away_N = I_N3[:,2]
    routes_N = I_N3[:,3]
    @assert model.M == 1
    routes_R4 = state["routes_R4"]

    #we have an additive mean so we must thin each point across its 3 components:
    #1: a frailty for origin airport
    #2: a farilty for dest airport
    #3: the distance*cluster mean

    post_shape_R = fill(model.a, model.R)
    post_rate_R = fill(model.b, model.R)

    post_shape1_T = fill(model.c, model.T)
    post_shape2_T = fill(model.c, model.T)
    post_rate1_T = fill(model.d, model.T)
    post_rate2_T = fill(model.d, model.T)

    Z1_NM = zeros(Int, model.N,1)
    Z2_NM = zeros(Int, model.N,1)
    Z_TTnew = zeros(Int, model.T, model.T)

    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    @views @threads for n in 1:model.N   
        home = home_N[n]
        away = away_N[n]
        D = D_R[routes_N[n]]
        mu = A_T[home]+B_T[away]+dist_NM[n,1]*U_R[routes_N[n]]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = sample_likelihood(model,mu,D)
        end
        if Y_NM[n, 1] > 0
            Z1_NM[n,1] = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, div(D,2)+1, Poisson(mu))
        end
    end


    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    Z_R3 = zeros(R,3)
    P_3 = zeros(3)
    @views for r in 1:R
        #access pre-calculated route info
        home = routes_R4[r,1]
        away = routes_R4[r,2]
        indices = routes_R4[r,3]
        Zr = sum(Z1_NM[indices,1])
        distance = routes_R4[r,4]
        numflights = length(indices)
        #thin
        if Zr > 0
            P_3 = vcat(A_T[home], B_T[away], distance*U_R[r])
            Z_R3[r, :] = rand(Multinomial(Zr, P_3 ./ sum(P_3)))
        end

        post_shape_R[r] += Z_R3[r,3]
        post_rate_R[r] += D_R[r]*distance*numflights
        #update frailty parameters
        post_shape1_T[home] += Z_R3[r,1]
        post_rate1_T[home] += D_R[r]*numflights
        post_shape2_T[away] += Z_R3[r,2]
        post_rate2_T[away] += D_R[r]*numflights
    end

    U_R = rand.(Gamma.(post_shape_R, 1 ./post_rate_R))

    A_T = rand.(Gamma.(post_shape1_T, 1 ./post_rate1_T))
    B_T = rand.(Gamma.(post_shape2_T, 1 ./post_rate2_T))

    #Update each D
    if isnothing(skipupdate) || !("D_R" in skipupdate)
        @views for r in 1:model.R
            home = routes_R4[r,1]
            away = routes_R4[r,2]
            indices = routes_R4[r,3]
            dist = routes_R4[r,4]
            logprob = 0
            mu = A_T[home]+B_T[away]+dist*U_R[r]
            
            logprobs = [logbinomial(Int((model.Dmax-1)/2),Int((d-1)/2)) + (d-1)*log(p)/2 + (model.Dmax - d)*log(1-p)/2 for d in 1:2:model.Dmax]
            for d in 1:2:model.Dmax
                for f in 1:length(indices)
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

    state = Dict("U_R" => U_R, "A_T"=>A_T, "B_T"=>B_T,"D_R"=>D_R,"p"=>p,
     "I_N3"=>I_N3, "dist_N" => dist_N, "routes_R4"=>routes_R4)
    return data, state
end