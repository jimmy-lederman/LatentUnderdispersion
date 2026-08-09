include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using LogExpFunctions

# STAR factor model with non-negative factor priors matching the order
# statistic models (referee 2, revision 2):
#   Y_ij = ceil(g_inv(Z_ij)) (0 if Z_ij < 0),  Z_ij ~ N(mu_ij, sigma2)
#   mu_ij = sum_k U_ik V_kj
#   U_:k ~ Dirichlet(a * 1_N),  V_kj ~ Gamma(shape c, rate d),  sigma2 ~ IG(a0, b0)
# Exact Gibbs updates require a == 1 (uniform-on-simplex) and c == 1
# (exponential), which is the referee-specified configuration
# theta_k ~ Dir(1), psi_kj ~ Ga(1, 0.01).
# parametric in the transformation functions so calls to model.g / model.g_inv
# dispatch statically in the hot loops (Function fields would box every call);
# same change as STARMF.jl, no numerical difference
struct STARMFNN{F1,F2} <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64   # Dirichlet concentration for columns of U_NK
    c::Float64   # gamma shape for V_KM
    d::Float64   # gamma rate for V_KM
    a0::Float64  # InverseGamma shape for sigma2
    b0::Float64  # InverseGamma scale for sigma2
    g::F1
    g_inv::F2
end

function STARlogpmf(model::STARMFNN, x, mu, sigma2)
    std = sqrt(sigma2)
    if x == 0
        z = (model.g(0) - mu) / std
        return logcdf(Normal(0, 1), z)
    else
        z1 = (model.g(x) - mu) / std
        z0 = (model.g(x - 1) - mu) / std
        lc1 = logcdf(Normal(0, 1), z1)
        lc0 = logcdf(Normal(0, 1), z0)
        if lc1 < lc0
            return lc0 + log1mexp(lc1 - lc0)
        else
            return lc1 + log1mexp(lc0 - lc1)
        end
    end
end

function evalulateLogLikelihood(model::STARMFNN, state, data, info, row, col)
    Y = data["Y_NM"][row, col]
    mu = dot(state["U_NK"][row, :], state["V_KM"][:, col])
    sigma2 = state["sigma2"]
    return STARlogpmf(model, Y, mu, sigma2)
end

function sample_prior(model::STARMFNN, info=nothing)
    dhyper = sample_rate_prior()   # hierarchical Gamma rate for V
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1 / dhyper), model.K, model.M)
    sigma2 = rand(InverseGamma(model.a0, model.b0))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2, "d" => dhyper)
    return state
end

function forward_sample(model::STARMFNN; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    sigma = sqrt(state["sigma2"])
    Mu_NM = state["U_NK"] * state["V_KM"]
    Z_NM = rand.(Normal.(Mu_NM, sigma))
    Y_NM = zeros(model.N, model.M)
    for n in 1:model.N
        for m in 1:model.M
            if Z_NM[n, m] < 0
                Y_NM[n, m] = 0
            else
                Y_NM[n, m] = ceil(model.g_inv(Z_NM[n, m]))
            end
        end
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

# Truncated draw from the density prop. to exp(-A*t^2/2 + B*t) on
# [lower, upper]. When A vanishes (possible only if the factor's loadings or
# weights are all zero) the density degenerates to exp(B*t): uniform for
# B == 0, truncated exponential otherwise.
function truncgauss(A, B, lower, upper)
    if A > 0 && isfinite(A) && isfinite(B)
        return rand(Truncated(Normal(B / A, sqrt(1 / A)), lower, upper))
    elseif B < 0
        return rand(Truncated(Exponential(-1 / B), lower, upper))
    elseif isfinite(upper)
        return lower + (upper - lower) * rand()
    else
        error("improper conditional in truncgauss: A=$A, B=$B on [$lower, Inf)")
    end
end

# --- Modular Gibbs updates (each leaves the joint invariant on its own) ---

# Impute heldout entries (in Y_NM, in place) and sample latent Z | Y, params
function update_Z!(model::STARMFNN, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    @views for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask) && mask[n, m] == 1
                z = rand(Normal(Mu_NM[n, m], sqrt(sigma2)))
                if z < 0
                    Y_NM[n, m] = 0
                else
                    Y_NM[n, m] = ceil(model.g_inv(z))
                end
            end
            if Y_NM[n, m] == 0
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sqrt(sigma2)), -Inf, model.g(0)))
            else
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sqrt(sigma2)), model.g(Y_NM[n, m] - 1), model.g(Y_NM[n, m])))
            end
        end
    end
end

# Gibbs update for each simplex column of U via 1-D mass-shift moves between
# a coordinate i1 and a donor i2: with s = U[i1,k] + U[i2,k] fixed and
# U[i1,k] = t, the conditional over t in [0, s] is a truncated normal
# (Dir(1) prior is flat on the simplex). Mu_NM kept in sync.
#
# mode = :classical (default) the classical simplex-Gibbs (Altmann et al.
#                   2014): eliminate coordinate N as the donor, scan i1=1:N-1
# mode = :random    N random (i1, i2) pairs per factor per call — empirically
#                   indistinguishable from :classical (compare_classical_U.jl)
function update_U!(model::STARMFNN, U_NK, V_KM, Z_NM, Mu_NM, sigma2; mode::Symbol=:classical)
    nmoves = mode == :classical ? model.N - 1 : model.N
    @views for k in 1:model.K
        for mv in 1:nmoves
            if mode == :classical
                i1 = mv
                i2 = model.N
            else
                i1 = rand(1:model.N)
                i2 = rand(1:(model.N - 1))
                if i2 >= i1
                    i2 += 1
                end
            end
            s = U_NK[i1, k] + U_NK[i2, k]
            if s <= 0
                continue
            end
            A = 0.0
            B = 0.0
            for j in 1:model.M
                v = V_KM[k, j]
                e = Z_NM[i1, j] - (Mu_NM[i1, j] - U_NK[i1, k] * v)
                w = Z_NM[i2, j] - (Mu_NM[i2, j] - U_NK[i2, k] * v) - s * v
                A += 2 * v^2
                B += v * (e - w)
            end
            A /= sigma2
            B /= sigma2
            t = truncgauss(A, B, 0.0, s)
            told1 = U_NK[i1, k]
            told2 = U_NK[i2, k]
            U_NK[i1, k] = t
            U_NK[i2, k] = s - t
            for j in 1:model.M
                Mu_NM[i1, j] += (t - told1) * V_KM[k, j]
                Mu_NM[i2, j] += ((s - t) - told2) * V_KM[k, j]
            end
        end
    end
end

# V update: Ga(1, d) prior = Exponential(d), conditional is a truncated
# normal on (0, Inf). Mu_NM kept in sync.
function update_V!(model::STARMFNN, U_NK, V_KM, Z_NM, Mu_NM, sigma2, drate=model.d)
    @views for k in 1:model.K
        for j in 1:model.M
            A = 0.0
            B = 0.0
            for i in 1:model.N
                u = U_NK[i, k]
                e = Z_NM[i, j] - (Mu_NM[i, j] - u * V_KM[k, j])
                A += u^2
                B += u * e
            end
            A /= sigma2
            B = B / sigma2 - drate
            vold = V_KM[k, j]
            vnew = truncgauss(A, B, 0.0, Inf)
            V_KM[k, j] = vnew
            for i in 1:model.N
                Mu_NM[i, j] += U_NK[i, k] * (vnew - vold)
            end
        end
    end
end

function update_sigma2(model::STARMFNN, Z_NM, Mu_NM)
    resid = Z_NM .- Mu_NM
    return rand(InverseGamma(model.a0 + model.N * model.M / 2,
                             model.b0 + sum(resid .^ 2) / 2))
end

function backward_sample(model::STARMFNN, data, state, mask=nothing)
    @assert model.a == 1 "exact simplex Gibbs update requires Dirichlet concentration a == 1"
    @assert model.c == 1 "exact truncated-normal update for V requires gamma shape c == 1"

    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    sigma2 = copy(state["sigma2"])
    Mu_NM = U_NK * V_KM
    Z_NM = zeros(model.N, model.M)

    dhyper = get(state, "d", model.d)
    update_Z!(model, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    update_U!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2)
    update_V!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, dhyper)
    sigma2 = update_sigma2(model, Z_NM, Mu_NM)
    dhyper = sample_rate_hyper(V_KM, model.c)   # d | V, conjugate

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2, "Z_NM" => Z_NM, "d" => dhyper)
    return data, state
end
