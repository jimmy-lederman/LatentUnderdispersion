# Hierarchical-mu variant of the STAR flights competitor -- the matched counterpart of
# flights_hier.jl. The same structural change is made to both models so the Table 2
# comparison stays like-for-like: a learned population distribution on the route-level
# location parameter, instead of a fixed one.
#
#   Z_i      ~ N(mu_route[i], sigma2_route[i]),   Y_i = ceil(g_inv(Z_i))
#   mu_k | m,s2 ~ N(m, s2)        <- population distribution, learned (was N(0, 50^2) fixed)
#   m        ~ N(0, tau02)        <- conjugate
#   s2       ~ InverseGamma(c0,d0)<- conjugate
#
# Note the flat STAR's mu_k ~ N(0, 50^2) is centred at zero, which is the more misspecified
# of the two starting points, so if either model gains from this change it is likelier to be
# this one.

include("flights_STAR.jl")

struct flights_STAR_hier{F1,F2} <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    a::Float64        # InverseGamma shape for the route variances sigma2_k
    b::Float64        # InverseGamma rate  for the route variances sigma2_k
    tau02::Float64    # prior variance for the population mean m
    c0::Float64       # InverseGamma shape for the population variance s2
    d0::Float64       # InverseGamma rate  for the population variance s2
    g::F1
    g_inv::F2
end

flights_STAR_hier(N, M, R, a, b, tau02, c0, d0, g, g_inv) =
    flights_STAR_hier{typeof(g),typeof(g_inv)}(N, M, R, a, b, tau02, c0, d0, g, g_inv)

function STARlogpmf(model::flights_STAR_hier, x, mu, sigma2)
    std = sqrt(sigma2)
    if x == 0
        return logcdf(Normal(0,1), (model.g_inv(0) - mu) / std)
    else
        lc1 = logcdf(Normal(0,1), (model.g(x) - mu) / std)
        lc0 = logcdf(Normal(0,1), (model.g(x - 1) - mu) / std)
        return lc1 < lc0 ? lc0 + log1mexp(lc1 - lc0) : lc1 + log1mexp(lc0 - lc1)
    end
end

function evalulateLogLikelihood(model::flights_STAR_hier, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    route = data["routes_N"][row]
    return STARlogpmf(model, Y, state["U_R"][route], state["sigma2_R"][route])
end

function sample_prior(model::flights_STAR_hier, info=nothing, constantint=nothing)
    m  = rand(Normal(0, sqrt(model.tau02)))
    s2 = rand(InverseGamma(model.c0, model.d0))
    U_R = rand(Normal(m, sqrt(s2)), model.R)
    sigma2_R = rand(InverseGamma(model.a, model.b), model.R)
    return Dict("U_R"=>U_R, "sigma2_R"=>sigma2_R, "m"=>m, "s2"=>s2)
end

function forward_sample(model::flights_STAR_hier; state=nothing, info=nothing)
    if isnothing(state); state = sample_prior(model, info); end
    U_R = state["U_R"]; sigma2_R = state["sigma2_R"]; routes_N = state["I_N3"][:,3]
    Y_NM = zeros(Int, model.N, model.M)
    for n in 1:model.N
        z = rand(Normal(U_R[routes_N[n]], sqrt(sigma2_R[routes_N[n]])))
        Y_NM[n,1] = z < 0 ? 0 : ceil(model.g_inv(z))
    end
    return Dict("Y_NM"=>Y_NM), state
end

function backward_sample(model::flights_STAR_hier, data, state, mask=nothing; skipupdate=nothing, skipupdatealways=nothing)
    Y_NM = data["Y_NM"]
    route_idx = data["route_idx"]::Vector{Vector{Int}}
    route_n = data["route_n"]::Vector{Int}
    routes_N = data["routes_N"]
    U_R = copy(state["U_R"])
    sigma2_R = copy(state["sigma2_R"])
    m = copy(state["m"])
    s2 = copy(state["s2"])
    @assert model.M == 1

    Z_NM = zeros(Float64, model.N, 1)
    g0 = model.g(0)
    @views @threads :static for n in 1:model.N
        r = routes_N[n]
        mu = U_R[r]; sd = sqrt(sigma2_R[r])
        if !isnothing(mask) && mask[n,1] == 1
            z = rand(Normal(mu, sd))
            Y_NM[n,1] = z < 0 ? 0 : ceil(model.g_inv(z))
        end
        y = Y_NM[n,1]
        Z_NM[n,1] = y == 0 ? rand(Truncated(Normal(mu, sd), -Inf, g0)) :
                             rand(Truncated(Normal(mu, sd), model.g(y - 1), model.g(y)))
    end

    # ---- mu_k | m, s2 : conjugate normal-normal, now shrunk toward the population mean ----
    @views for r in 1:model.R
        indices = route_idx[r]; n = route_n[r]; sigma2 = sigma2_R[r]
        z_N = Z_NM[indices,1]; s = sum(z_N)
        V = 1 / (n / sigma2 + 1 / s2)
        mu_post = V * (s / sigma2 + m / s2)
        U_R[r] = rand(Normal(mu_post, sqrt(V)))
        s2r = sum((z_N .- U_R[r]).^2)
        sigma2_R[r] = rand(InverseGamma(model.a + n/2, model.b + s2r/2))
    end

    # ---- population mean m | mu, s2 : conjugate ----
    Vm = 1 / (model.R / s2 + 1 / model.tau02)
    m = rand(Normal(Vm * (sum(U_R) / s2), sqrt(Vm)))

    # ---- population variance s2 | mu, m : conjugate ----
    s2 = rand(InverseGamma(model.c0 + model.R/2, model.d0 + sum((U_R .- m).^2)/2))

    return data, Dict("U_R"=>U_R, "sigma2_R"=>sigma2_R, "m"=>m, "s2"=>s2)
end
