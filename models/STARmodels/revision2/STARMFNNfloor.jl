# STAR-NN with floor-style binning: Y = y  <=>  Z in (g(y), g(y+1)], zero bin
# (-inf, g(1)]. Removes the P(Y=0) <= 1/2 ceiling that ceil-style binning
# (zero bin (-inf, g(0)=0]) imposes when the latent mean is constrained
# non-negative: here P(Y=0) = Phi((g(1)-mu)/sigma) -> 1 as mu -> 0.
# Everything else (priors, U/V/sigma2 updates) is identical to STARMFNN.
include("STARMFNN.jl")

# parametric in g/g_inv for static dispatch in the hot loops (see STARMFNN.jl)
struct STARMFNNF{F1,F2} <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    c::Float64
    d::Float64
    a0::Float64
    b0::Float64
    g::F1
    g_inv::F2
end

function STARlogpmf(model::STARMFNNF, x, mu, sigma2)
    std = sqrt(sigma2)
    if x == 0
        return logcdf(Normal(0, 1), (model.g(1) - mu) / std)
    else
        lc1 = logcdf(Normal(0, 1), (model.g(x + 1) - mu) / std)
        lc0 = logcdf(Normal(0, 1), (model.g(x) - mu) / std)
        if lc1 < lc0
            return lc0 + log1mexp(lc1 - lc0)
        else
            return lc1 + log1mexp(lc0 - lc1)
        end
    end
end

function evalulateLogLikelihood(model::STARMFNNF, state, data, info, row, col)
    Y = data["Y_NM"][row, col]
    mu = dot(state["U_NK"][row, :], state["V_KM"][:, col])
    return STARlogpmf(model, Y, mu, state["sigma2"])
end

function sample_prior(model::STARMFNNF, info=nothing)
    dhyper = sample_rate_prior()   # hierarchical Gamma rate for V
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1 / dhyper), model.K, model.M)
    sigma2 = rand(InverseGamma(model.a0, model.b0))
    return Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2, "d" => dhyper)
end

function ytoz_floor(model::STARMFNNF, z)
    # Y = y <=> Z in (g(y), g(y+1)]; z <= g(1) -> 0
    z <= model.g(1) && return 0.0
    return ceil(model.g_inv(z)) - 1
end

function forward_sample(model::STARMFNNF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Z_NM = rand.(Normal.(Mu_NM, sqrt(state["sigma2"])))
    Y_NM = zeros(model.N, model.M)
    for n in 1:model.N, m in 1:model.M
        Y_NM[n, m] = ytoz_floor(model, Z_NM[n, m])
    end
    return Dict("Y_NM" => Y_NM), state
end

function update_Z_floor!(model::STARMFNNF, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    @views for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask) && mask[n, m] == 1
                z = rand(Normal(Mu_NM[n, m], sqrt(sigma2)))
                Y_NM[n, m] = ytoz_floor(model, z)
            end
            if Y_NM[n, m] == 0
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sqrt(sigma2)), -Inf, model.g(1)))
            else
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sqrt(sigma2)),
                                            model.g(Y_NM[n, m]), model.g(Y_NM[n, m] + 1)))
            end
        end
    end
end

function backward_sample(model::STARMFNNF, data, state, mask=nothing)
    @assert model.a == 1 && model.c == 1
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    sigma2 = copy(state["sigma2"])
    Mu_NM = U_NK * V_KM
    Z_NM = zeros(model.N, model.M)

    dhyper = get(state, "d", model.d)
    update_Z_floor!(model, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    # U/V/sigma2 conditionals depend only on (Z, Mu): reuse a shim STARMFNN
    shim = STARMFNN(model.N, model.M, model.K, model.a, model.c, model.d,
                    model.a0, model.b0, model.g, model.g_inv)
    update_U!(shim, U_NK, V_KM, Z_NM, Mu_NM, sigma2)
    update_V!(shim, U_NK, V_KM, Z_NM, Mu_NM, sigma2, dhyper)
    sigma2 = update_sigma2(shim, Z_NM, Mu_NM)
    dhyper = sample_rate_hyper(V_KM, model.c)   # d | V, conjugate

    return data, Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2, "Z_NM" => Z_NM, "d" => dhyper)
end
