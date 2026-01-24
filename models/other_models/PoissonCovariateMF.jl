include("../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra

struct PoissonCovariateMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64
    h::Float64

end

function evalulateLogLikelihood(model::PoissonCovariateMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    dist = info["dist_NM"][row,col]
    mu = dot(state["U_NK1"][row,:], state["V_KM1"][:,col]) + dist*dot(state["U_NK2"][row,:], state["V_KM2"][:,col])
    return logpdf(Poisson(mu), Y)
end

function sample_prior(model::PoissonCovariateMF, info=nothing)
    U_NK1 = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM1 = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    U_NK2 = rand(Gamma(model.e, 1/model.f), model.N, model.K)
    V_KM2 = rand(Gamma(model.g, 1/model.h), model.K, model.M)
    state = Dict("U_NK1" => U_NK1, "V_KM1" => V_KM1, "U_NK2" => U_NK2, "V_KM2" => V_KM2,
    "dist_NM"=>info["dist_NM"]);
    return state
end

function forward_sample(model::PoissonCovariateMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM1 = state["U_NK1"] * state["V_KM1"]
    Mu_NM2 = state["U_NK2"] * state["V_KM2"]
    dist_NM = info["dist_NM"]
    Y_NM = rand.(Poisson.(Mu_NM1 + Mu_NM2 .* dist_NM))
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::PoissonCovariateMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK1 = copy(state["U_NK1"])
    V_KM1 = copy(state["V_KM1"])
    U_NK2 = copy(state["U_NK2"])
    V_KM2 = copy(state["V_KM2"])
    dist_NM = copy(state["dist_NM"])
    Y_NMK = zeros(model.N, model.M, 2*model.K)
    P_2K = zeros(2*model.K)
    if !isnothing(mask)
        Mu_NM1 = U_NK1 * V_KM1
        Mu_NM2 = U_NK2 * V_KM2
    end
    # Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(Poisson(Mu_NM1[n,m] + dist_NM[n,m]*Mu_NM2[n,m]))
                end
            end
            if Y_NM[n, m] > 0
                P_2K[1:model.K] = U_NK1[n, :] .* V_KM1[:, m]
                P_2K[(model.K+1):end] = dist_NM[n,m] * U_NK2[n, :] .* V_KM2[:, m]
                P_2K[:] = P_2K / sum(P_2K)
                Y_NMK[n, m, :] = rand(Multinomial(Y_NM[n, m], P_2K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Y_NMK[n, :, k])
            post_rate = model.b + sum(V_KM1[k, :])
            U_NK1[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:model.K
            post_shape = model.c + sum(Y_NMK[:, m, k])
            post_rate = model.d + sum(U_NK1[:, k])
            V_KM1[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.e + sum(Y_NMK[n, :, model.K + k]) 
            post_rate = model.f + sum(V_KM2[k, :].*dist_NM[n,:])
            U_NK2[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:model.K
            post_shape = model.g + sum(Y_NMK[:, m, model.K + k]) 
            post_rate = model.h + sum(U_NK2[:, k].*dist_NM[:,m])
            V_KM2[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end




    state = Dict("U_NK1" => U_NK1, "V_KM1" => V_KM1, "U_NK2" => U_NK2, "V_KM2" => V_KM2,
     "dist_NM"=>dist_NM);
    return data, state
end

