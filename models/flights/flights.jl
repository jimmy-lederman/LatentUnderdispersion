include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
using Distributions
using Base.Threads

struct flights <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64
    D::Int64
    j::Int64
    dist::Function
end

function sampleCRT(Y,R)
    # if Y > 10000
    #     println(Y)
    # end
    if Y == 0
        return 0
    elseif Y == 1
        probs = [1]
    else
        probs = vcat([1],[R/(R+i-1) for i in 2:Y])
    end
    return sum(rand.(Bernoulli.(probs)))
end

function evalulateLogLikelihood(model::flights, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]
    dist = info["dist_NM"]

    A_T = state["A_T"]
    B_T = state["B_T"]

    mu = A_T[home] + B_T[away] + dist[row,1]*state["U_K"][state["Z_TT"][home,away]]
    if isnothing(state["p"])
        if model.D == 1
            return logpdf(model.dist(mu), Y)
        else
            try
                return logpdf(OrderStatistic(mode.dist(mu), model.D, model.j), Y)
            catch ex #this needs to be updated with the non symbolic version
                @assert model.D == model.j
                return logpmfMaxPoisson(Y,mu,model.D)
            end
        end
    else
        p = state["p"]
        if model.D == 1
            return logpdf(model.dist(mu,1-p), Y)
        else
            return logpdf(OrderStatistic(model.dist(mu,1-p), model.D, model.j), Y)
        end
    end
end



function sample_likelihood(model::flights, mu,p=nothing,n=1)
    if isnothing(p)
        dist = model.dist(mu)
    else
        dist = model.dist(mu,1-p)
    end
    if model.D == 1
        return rand(dist,n)
    else
        return rand(OrderStatistic(dist, model.D, model.j),n)
    end
end

function sample_prior(model::flights, info=nothing)
    p = nothing
    Z_TT = rand(Categorical(model.K), model.T, model.T)
    U_K = rand(Gamma(model.a, 1/model.b), model.K)
    A_T = rand(Gamma(model.c, 1/model.d), model.T)
    B_T = rand(Gamma(model.c, 1/model.d), model.T)
    @assert !isnothing(info)
    try
        dist = model.dist(1)
    catch e 
        p = rand(Beta(model.alpha, model.beta))
    end

    state = Dict("Z_TT" => Z_TT, "U_K" => U_K, "A_T" => A_T, "B_T" => B_T,
                "I_NM"=>info["I_NM"], "dist_NM" => info["dist_NM"], "routes_R4"=>nothing,
                "p"=>p)
    return state
end

function forward_sample(model::flights; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    Z_TT = state["Z_TT"]
    U_K = state["U_K"]

    A_T = state["A_T"]
    B_T = state["B_T"]

    I_NM = state["I_NM"]
    dist_NM = state["dist_NM"]

    p = state["p"]
    
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        mu = A_T[home] + B_T[away] + dist_NM[n,1]*U_K[Z_TT[home, away]]
        # println(mu)
        Y_NM[n,1] = sample_likelihood(model,mu,p)
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

function backward_sample(model::flights, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    Z_TT = copy(state["Z_TT"])
    U_K = copy(state["U_K"])
    I_NM = copy(state["I_NM"])
    A_T = copy(state["A_T"])
    B_T = copy(state["B_T"])
    p = state["p"]
    dist_NM = copy(state["dist_NM"])
    info = Dict("I_NM"=>I_NM,"dist_NM"=>dist_NM)

    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 1

    if !haskey(state, "routes_R4") || isnothing(state["routes_R4"])
        routes_R4 = Vector{Any}()
        for t1 in 1:model.T
            for t2 in 1:model.T
                bitvector = home_N .== t1 .&& away_N .== t2
                if sum(bitvector) != 0
                    indices = findall(bitvector)
                    distances = dist_NM[indices, 1]
                    @assert all(x -> x == distances[1], distances) "Not all distances are equal"
                    push!(routes_R4, [t1,t2,indices, distances[1]])
                end
            end
        end
        routes_R4 = hcat(routes_R4...)'
    else
        routes_R4 = state["routes_R4"]
    end

    R = size(routes_R4)[1]

    #we have an additive mean so we must thin each point across its 3 components:
    #1: a frailty for origin airport
    #2: a farilty for dest airport
    #3: the distance*cluster mean

    post_shape_K = fill(model.a, model.K)
    post_rate_K = fill(model.b, model.K)

    post_shape1_T = fill(model.c, model.T)
    post_shape2_T = fill(model.c, model.T)
    post_rate1_T = fill(model.d, model.T)
    post_rate2_T = fill(model.d, model.T)

    Z1_NM = zeros(Int, model.N,1)
    Z2_NM = zeros(Int, model.N,1)
    Z_TTnew = zeros(Int, model.T, model.T)

    if isnothing(p)
        lik = model.dist
    else
        lik = x -> model.dist(x,1-p)
    end

    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    # @views @threads for n in 1:model.N
    @views @threads for n in 1:model.N   
        home = home_N[n]
        away = away_N[n]
        if !isnothing(mask) && mask[n,1] == 1
            mu = A_T[home]+B_T[away]+dist_NM[n,1]*U_K[Z_TT[home,away]]
            Y_NM[n,1] = sample_likelihood(model,mu,p)
        end
        if Y_NM[n, 1] > 0 || model.D != model.j
            mu = A_T[home] + B_T[away] + U_K[Z_TT[home,away]]*dist_NM[n,1]
            Z1_NM[n,1] = sampleSumGivenOrderStatistic(Y_NM[n, 1], model.D, model.j, lik(mu))
            if lik(mu) isa NegativeBinomial
                Z2_NM[n,1] = copy(Z1_NM[n,1])
                Z1_NM[n,1] = sampleCRT(Z1_NM[n,1], model.D*mu)
            end
        end
    end

    #update p if necessary
    if lik(1) isa NegativeBinomial
        mu_all = 0
        @views for n in 1:model.N
            home = home_N[n]
            away = away_N[n]
            mu_all += A_T[home]+B_T[away]+dist_NM[n,1]*U_K[Z_TT[home,away]]
        end
        post_alpha = model.alpha + sum(Z2_NM)
        post_beta = model.beta + model.D*mu_all
        p2 = copy(rand(Beta(post_alpha,post_beta)))
        rate_factor = model.D*log(1/(1-p2))
    else
        p2 = nothing
        rate_factor = model.D
    end

    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    Z_R3 = zeros(R,3)
    P_3 = zeros(3)
    #locker = Threads.SpinLock()
    @views for r in 1:R
    #@views @threads for r in 1:R
        #access pre-calculated route info
        home = routes_R4[r,1]
        away = routes_R4[r,2]
        indices = routes_R4[r,3]
        Zr = sum(Z1_NM[indices,1])
        distance = routes_R4[r,4]
        numflights = length(indices)
        #thin
        if Zr > 0
            P_3 = vcat(A_T[home], B_T[away], distance*U_K[Z_TT[home,away]])
            # P_3[1] = A_T[home]
            # P_3[2] = B_T[away]
            # P_3[3] = distance*U_K[Z_TT[home,away]]
            #P_3 = P_3 ./ sum(P_3)
            Z_R3[r, :] = rand(Multinomial(Zr, P_3 ./ sum(P_3)))
        end

        #calc categorical probabilities
        logprobvec_K = zeros(model.K)
        for u in 1:model.K
            logprobvec_K[u] = logpdf(Poisson(rate_factor*numflights*U_K[u]*distance), Z_R3[r,3])
        end
        #sample categorical using Gumbel trick
        g_K = rand(Gumbel(0,1), model.K)
        Z_TTnew[home,away] = argmax(g_K + logprobvec_K)
        #lock(locker)
        #update cluster mean parameters
        post_shape_K[Z_TTnew[home,away]] += Z_R3[r,3]
        post_rate_K[Z_TTnew[home,away]] += rate_factor*distance*numflights
        #update frailty parameters
        post_shape1_T[home] += Z_R3[r,1]
        post_rate1_T[home] += rate_factor*numflights
        post_shape2_T[away] += Z_R3[r,2]
        post_rate2_T[away] += rate_factor*numflights
        #unlock(locker)
    end

    U_K = rand.(Gamma.(post_shape_K, 1 ./post_rate_K))

    A_T = rand.(Gamma.(post_shape1_T, 1 ./post_rate1_T))
    B_T = rand.(Gamma.(post_shape2_T, 1 ./post_rate2_T))
    flush(stdout)
    state = Dict("Z_TT" => Z_TTnew, "U_K" => U_K, "A_T"=>A_T, "B_T"=>B_T,
     "I_NM"=>I_NM, "dist_NM" => dist_NM, "routes_R4"=>routes_R4,
     "p"=>p2)
    return data, state
end