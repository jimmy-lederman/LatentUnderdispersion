include("../helper/MatrixMF.jl")
using Distributions
using Base.Threads

struct PoissonMixture <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function evalulateLogLikelihood(model::PoissonMixture, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    home = info["I_NM"][row,1]
    away = info["I_NM"][row,2]

    Z_TT = state["Z_TT"]
    U_K = state["U_K"]
    A_T = state["A_T"]
    B_T = state["B_T"]

    mu = A_T[home] + B_T[away] + dist_NM[row,1]*U_K[Z_TT[home,away]]
   
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonMixture, info=nothing)
    Z_TT = rand(Categorical(model.K), model.T, model.T)
    U_K = rand(Gamma(model.a, 1/model.b), model.K)
    A_T = rand(Gamma(model.c, 1/model.d), model.T)
    B_T = rand(Gamma(model.c, 1/model.d), model.T)
    @assert !isnothing(info)
    state = Dict("Z_TT" => Z_TT, "U_K" => U_K, "A_T" => A_T, "B_T" => B_T,
    "I_NM"=>info["I_NM"], "dist_NM" => info["dist_NM"], "routes_R4"=>nothing)
    return state
end

function forward_sample(model::PoissonMixture; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end

    Z_TT = state["Z_TT"]
    U_K = state["U_K"]

    A_T = state["A_T"]
    B_T = state["B_T"]

    I_NM = state["I_NM"]
    dist_NM = state["dist_NM"]
    
    home_N = state["I_NM"][:,1]
    away_N = state["I_NM"][:,2]

    @assert model.M == 1
    Y_NM = zeros(model.N,model.M)
    for n in 1:model.N
        home = home_N[n]
        away = away_N[n]
        mu = A_T[home] + B_T[away] + dist_NM[n,1]*U_K[Z_TT[home, away]]
        Y_NM[n,1] = rand(Poisson(mu))
    end
    data = Dict("Y_NM" => Int.(Y_NM))
    return data, state
end

function backward_sample(model::PoissonMixture, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    Z_TT = copy(state["Z_TT"])
    U_K = copy(state["U_K"])
    I_NM = copy(state["I_NM"])
    A_T = copy(state["A_T"])
    B_T = copy(state["B_T"])
    dist_NM = copy(state["dist_NM"])
    home_N = I_NM[:,1]
    away_N = I_NM[:,2]
    @assert model.M == 1

    #create route summary of data
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

    if !isnothing(mask) 
        @views for n in 1:model.N
            if mask[n,1] == 1
                Y_NM[n,1] = rand(Poisson(A_T[home]+B_T[away]+dist_NM[n,1]*U_K[Z_TT[home,away]]))
            end
        end
    end

    post_shape_K = fill(model.a, model.K)
    post_rate_K = fill(model.b, model.K)

    post_shape1_T = fill(model.c, model.T)
    post_shape2_T = fill(model.c, model.T)
    post_rate1_T = fill(model.d, model.T)
    post_rate2_T = fill(model.d, model.T)

    #we will refence the old Z_TT when we thin
    #but calculate the new ones and put them in here
    Z_TTnew = zeros(Int, model.T, model.T)

    Y_R3 = zeros(R,3)
    @views @threads for r in 1:R
        #access pre-calculated route info
        home = routes_R4[r,1]
        away = routes_R4[r,2]
        indices = routes_R4[r,3]
        Yr = sum(Y_NM[indices,1])
        distance = routes_R4[r,4]
        numflights = length(indices)
        #thin
        if Yr > 0
            # P_3[1] = A_T[home]
            # P_3[2] = B_T[away]
            # P_3[3] = distance*U_K[Z_TT[home,away]]
            P_3 = vcat(A_T[home], B_T[away],  distance*U_K[Z_TT[home,away]])
            Y_R3[r, :] = rand(Multinomial(Yr, P_3 ./ sum(P_3)))
        end

        #calc categorical probabilities
        logprobvec_K = zeros(model.K)
        for u in 1:model.K
            logprobvec_K[u] = logpdf(Poisson(numflights*U_K[u]*distance), Y_R3[r,3])
        end
        #sample categorical using Gumbel trick
        g_K = rand(Gumbel(0,1), model.K)
        Z_TTnew[home,away] = argmax(g_K + logprobvec_K)

        lock(locker)
        #update cluster mean parameters
        post_shape_K[Z_TTnew[home,away]] += Y_R3[r,3]
        post_rate_K[Z_TTnew[home,away]] += distance*numflights
        #update frailty parameters
        post_shape1_T[home] += Y_R3[r,1]
        post_rate1_T[home] += numflights
        post_shape2_T[away] += Y_R3[r,2]
        post_rate2_T[away] += numflights
        unlock(locker)
    end

    U_K[:] = rand.(Gamma.(post_shape_K, 1 ./post_rate_K))

    A_T[:] = rand.(Gamma.(post_shape1_T, 1 ./post_rate1_T))
    B_T[:] = rand.(Gamma.(post_shape2_T, 1 ./post_rate2_T))

    state = Dict("Z_TT" => Z_TTnew, "U_K" => U_K, "A_T"=>A_T, "B_T"=>B_T,
     "I_NM"=>I_NM, "dist_NM" => dist_NM, "routes_R4"=>routes_R4)

    return data, state
end