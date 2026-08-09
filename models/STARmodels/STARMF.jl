include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using LinearAlgebra
using Base.Threads

# parametric in the transformation functions so calls to model.g / model.g_inv
# dispatch statically in the hot loops (Function fields would box every call)
struct STARMF{F1,F2} <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    # tau_phi2::Float64
    # tau_theta2::Float64
    g::F1
    g_inv::F2
end

function STARlogpmf(model::STARMF, x, mu, sigma2)
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

function evalulateLogLikelihood(model::STARMF, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    sigma2 = state["sigma2"]

    x =  STARlogpmf(model,Y,mu,sigma2)
    return x
end

function sample_prior(model::STARMF,info=nothing)
    sigma2U = rand(InverseGamma(model.a, model.b))
    sigma2V = rand(InverseGamma(model.a, model.b))
    U_NK = rand(Normal(0, sqrt(sigma2U)), model.N, model.K)
    V_KM = rand(Normal(0, sqrt(sigma2V)), model.K, model.M)
    sigma2 = rand(InverseGamma(model.a, model.b))
    
    
    
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM,
     "sigma2"=>sigma2, "sigma2U"=>sigma2U, "sigma2V"=>sigma2V)
    return state
end

function forward_sample(model::STARMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    sigma2 = copy(state["sigma2"])
    sigma_NM = fill(sqrt(sigma2), model.N, model.M)
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NM = rand.(Normal.(U_NK*V_KM, sigma_NM))
    Y_NM = zeros(model.N, model.M)
    #Z_NM = state["Z_NM"] 
    for n in 1:model.N
        for m in 1:model.M
            if Z_NM[n,m] < 0
                Y_NM[n,m] = 0
            else
                Y_NM[n,m] = ceil(model.g_inv(Z_NM[n,m]))
            end
        end
    end


    data = Dict("Y_NM" => Y_NM)
    #state = Dict("U_NK" => state["U_NK"], "V_KM" => state["V_KM"])
    return data, state 
end

# U and V updates, in their own functions with the fitted mean Mu carried
# alongside and patched in O(M) / O(N) after each draw.
#
# The previous form rebuilt the full row/column mean inside the innermost loop
#     R = Z[n,:] .- vec(U[n,:]' * V) .+ vec(U[n,k] * V[k,:])
# which allocated three temporaries per (n,k) -- 0.30 MB of garbage per Gibbs
# sweep -- and, being inlined in a long function body, was not specialized.
# Same algebra and the same RNG draws in the same order; the arithmetic is
# reassociated, so states agree to rounding rather than bitwise
# (revision2_experiments/perf_review/validate_starmf_fair.jl).
function starmf_update_U!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, sigma2U)
    @inbounds for n in 1:model.N
        for k in 1:model.K
            vk2 = 0.0
            vr = 0.0
            for m in 1:model.M
                v = V_KM[k, m]
                vk2 += v * v
                vr += v * (Z_NM[n, m] - Mu_NM[n, m] + U_NK[n, k] * v)
            end
            sigma2_phi = 1 / (vk2 / sigma2 + 1 / sigma2U)
            mu_phi = sigma2_phi * vr / sigma2
            uold = U_NK[n, k]
            unew = rand(Normal(mu_phi, sqrt(sigma2_phi)))
            U_NK[n, k] = unew
            d = unew - uold
            for m in 1:model.M
                Mu_NM[n, m] += d * V_KM[k, m]
            end
        end
    end
    return nothing
end

function starmf_update_V!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, sigma2V)
    @inbounds for k in 1:model.K
        for m in 1:model.M
            uk2 = 0.0
            ur = 0.0
            for n in 1:model.N
                u = U_NK[n, k]
                uk2 += u * u
                ur += u * (Z_NM[n, m] - Mu_NM[n, m] + u * V_KM[k, m])
            end
            sigma2_theta = 1 / (uk2 / sigma2 + 1 / sigma2V)
            mu_theta = sigma2_theta * ur / sigma2
            vold = V_KM[k, m]
            vnew = rand(Normal(mu_theta, sqrt(sigma2_theta)))
            V_KM[k, m] = vnew
            d = vnew - vold
            for n in 1:model.N
                Mu_NM[n, m] += U_NK[n, k] * d
            end
        end
    end
    return nothing
end

function backward_sample(model::STARMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    #Z_NM = copy(state["Z_NM"])
    # Z_NMK = zeros(Int, model.N, model.M, model.K)
    #Z_NMK = copy(state["Z_NMK"])
    Mu_NM = U_NK * V_KM

    sigma2 = copy(state["sigma2"])
    sigma2U = copy(state["sigma2U"])
    sigma2V = copy(state["sigma2V"])
    Z_NM = zeros(model.N,model.M)

    sd = sqrt(sigma2)   # constant across the loop; hoisted (identical value)
    g0 = model.g(0)
    @views @threads for n in 1:model.N
        @views for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    z = rand(Normal(Mu_NM[n,m], sd))
                    if z < 0
                        Y_NM[n,m] = 0
                    else
                        Y_NM[n,m] = ceil(model.g_inv(z))
                    end
                end
            end
            if Y_NM[n,m] == 0
                Z_NM[n,m] = rand(Truncated(Normal(Mu_NM[n,m], sd), -Inf, g0))
            else
                Z_NM[n,m] = rand(Truncated(Normal(Mu_NM[n,m], sd), model.g(Y_NM[n,m] - 1), model.g(Y_NM[n,m])))
            end
        end
    end

    # Update Phi
    starmf_update_U!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, sigma2U)

    # Update Theta
    starmf_update_V!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, sigma2V)

    # Update sigma^2
    # post_a = model.a + model.N/2
    # post_b = model.b + sum((Z_N .- mu).^2)/2
    # sigma2 = rand(InverseGamma(post_a, post_b))
    ss = 0.0
    @inbounds for i in eachindex(Z_NM)   # Mu is maintained, so no rebuild here
        r = Z_NM[i] - Mu_NM[i]
        ss += r * r
    end
    sigma2 = rand(InverseGamma(model.a + model.N*model.M/2,
                               model.b + ss/2))

    # Update sigma2U
    sigma2U = rand(InverseGamma(model.a + model.N * model.K / 2,
                                model.b + sum(U_NK.^2)/2))

    # Update sigma2V
    sigma2V = rand(InverseGamma(model.a + model.K * model.M / 2,
                                model.b + sum(V_KM.^2)/2))

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2"=>sigma2, 
    "Z_NM"=>Z_NM, "sigma2U"=>sigma2U, "sigma2V"=>sigma2V)
    return data, state
end
