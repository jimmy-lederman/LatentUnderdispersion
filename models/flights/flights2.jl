include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    R::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    alpha::Float64
    beta::Float64t
end

function sampleCRT(Y,R)
    # if Y > 10000
    #     println(Y)
    # end
    if Y == 0
        return 0
    elseif Y == 1
        return 1
    else
        probs = [R/(R+i-1) for i in 2:Y]
    end
    return 1 + sum(rand.(Bernoulli.(probs)))
end

# function evalulateLogLikelihood(model::flights, state, data, info, row, col)
#     Y = data["Y_NM"][row,col]
#     @assert size(data["Y_NM"])[2] == 1
#     home = info["I_NM"][row,1]
#     away = info["I_NM"][row,2]
#     dist = info["dist_NM"]

#     A_T = state["A_T"]
#     B_T = state["B_T"]

#     mu = A_T[home] + B_T[away] + dist[row,1]*state["U_K"][state["Z_TT"][home,away]]
#     if isnothing(state["p"])
#         if model.D == 1
#             return logpdf(model.dist(mu), Y)
#         else
#             @assert model.D == model.j
#             return logpmfMaxPoisson(Y,mu,model.D)
#         end
#     else
#         p = state["p"]
#         if model.D == 1
#             return logpdf(model.dist(mu,1-p), Y)
#         else
#             return logpdf(OrderStatistic(model.dist(mu,1-p), model.D, model.j), Y)
#         end
#     end
# end



function sample_likelihood(model::flights, mu,D,n=1)
    if D == 1
        if n == 1
            return rand(Poisson(mu))
        else
            return rand(Poisson(mu),n)
        end
    else
        if n == 1
            return rand(OrderStatistic(Poisson(mu), D, D))
        else
            return rand(OrderStatistic(Poisson(mu), D, D),n)
        end
    end
end

function sample_prior(model::flights, info=nothing)
    p = nothing
    Z_TT = rand(Categorical(model.K), model.T, model.T)
    U_K = rand(Gamma(model.a, 1/model.b), model.K)
    A_T = rand(Gamma(model.c, 1/model.d), model.T)
    B_T = rand(Gamma(model.c, 1/model.d), model.T)
    p = rand(Beta(model.alpha,model.beta))
    #@assert !isnothing(info)
    D_R = rand(Binomial(model.Dmax, p), model.R)


    state = Dict("Z_TT" => Z_TT, "U_K" => U_K, "A_T" => A_T, "B_T" => B_T, "D_R"=>D_R, "p"=>p,
                "I_N3"=>info["I_N3"], "dist_N" => info["dist_N"], "routes_R4"=>info["routes_R4"],
                )
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
        mu = A_T[home] + B_T[away] + dist_NM[n,1]*U_K[Z_TT[home, away]]
        D = D_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,D)
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
    routes_N = I_NM[:,3]
    @assert model.M == 1

    #we are gonnaassume we have computed this ahead of time
    routes_R4 = state["routes_R4"]
    #each row is origin, dest, a list of the indices of data in that route, and the distance
    # if !haskey(state, "routes_R4") || isnothing(state["routes_R4"])
    #     routes_R4 = Vector{Any}()
    #     for t1 in 1:model.T
    #         for t2 in 1:model.T
    #             bitvector = home_N .== t1 .&& away_N .== t2
    #             if sum(bitvector) != 0
    #                 indices = findall(bitvector)
    #                 distances = dist_NM[indices, 1]
    #                 @assert all(x -> x == distances[1], distances) "Not all distances are equal"
    #                 push!(routes_R4, [t1,t2,indices, distances[1]])
    #             end
    #         end
    #     end
    #     routes_R4 = hcat(routes_R4...)'
    # else
    #     routes_R4 = state["routes_R4"]
    # end

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

    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    # @views @threads for n in 1:model.N
    @views @threads for n in 1:model.N   
        home = home_N[n]
        away = away_N[n]
        D = D_R[routes_N[n]]
        mu = A_T[home]+B_T[away]+dist_NM[n,1]*U_K[Z_TT[home,away]]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = sample_likelihood(model,mu,D)
        end
        if Y_NM[n, 1] > 0
            Z1_NM[n,1] = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, D, Poisson(mu))
        end
    end

    #Update each D
    @views for r in 1:model.R
        home = routes_R4[r,1]
        away = routes_R4[r,2]
        indices = routes_R4[r,3]
        dist = routes_R4[r,4]
        logprob = 0
        mu = A_T[home]+B_T[away]+dist*U_K[Z_TT[home,away]]
        
        logprobs = [logbinomial(model.Dmax-1,d-1) + (d-1)*log(p) + (model.Dmax - d)*log(1-p) for d in 1:model.Dmax]
        for d in 1:model.Dmax
            for f in 1:length(indices)
                Y = Y_NM[indices[f], 1]
                logprobs[d] += logpmfMaxPoisson(Y,mu,d)
            end
        end
        D_R = argmax(rand(Gumbel(0,1)) + logprobs)
    end

    #update p,
    p = rand(Beta(model.alpha + sum(D_R)- model.R, model.beta + model.Dmax*R - sum(D_R)))
    #might be wrong

    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    Z_R3 = zeros(R,3)
    P_3 = zeros(3)
    #locker = Threads.SpinLock()
    logprobvec_RK = zeros(R,model.K)
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
            logprobvec_K[u] = logpdf(Poisson(D_R[r]*numflights*U_K[u]*distance), Z_R3[r,3])
        end
        logprobvec_RK[r,:] =  logprobvec_K
        #sample categorical using Gumbel trick
        g_K = rand(Gumbel(0,1), model.K)
        Z_TTnew[home,away] = argmax(g_K + logprobvec_K)
        #lock(locker)
        #update cluster mean parameters
        post_shape_K[Z_TTnew[home,away]] += Z_R3[r,3]
        post_rate_K[Z_TTnew[home,away]] += D_R[r]*distance*numflights
        #update frailty parameters
        post_shape1_T[home] += Z_R3[r,1]
        post_rate1_T[home] += D_R[r]*numflights
        post_shape2_T[away] += Z_R3[r,2]
        post_rate2_T[away] += D_R[r]*numflights

    end

    U_K = rand.(Gamma.(post_shape_K, 1 ./post_rate_K))

    A_T = rand.(Gamma.(post_shape1_T, 1 ./post_rate1_T))
    B_T = rand.(Gamma.(post_shape2_T, 1 ./post_rate2_T))
    flush(stdout)
    state = Dict("Z_TT" => Z_TTnew, "U_K" => U_K, "A_T"=>A_T, "B_T"=>B_T,
     "I_NM"=>I_NM, "dist_NM" => dist_NM, "routes_R4"=>routes_R4,"logprobvec_RK"=>logprobvec_RK,
     "p"=>p2)
    return data, state
end