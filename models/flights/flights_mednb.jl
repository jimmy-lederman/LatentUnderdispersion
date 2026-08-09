# Median-of-D Negative Binomial variant of the flights model.
#
#   Y_i        ~ median of D_k iid NB(r_k, p_k),   k = route[i]
#   r_k | b    ~ Gamma(a, rate b),  b ~ Gamma(a0, b0)     (as in flights_hier.jl)
#   p_k        ~ Beta(alpha_p, beta_p)
#   D_k        ~ OddBinomial(Dmax, rho),  rho ~ Beta(alpha, beta)
#
# MOTIVATION (revision2_experiments/flights/why_star_wins.jl): with a Poisson parent the
# achievable dispersion is a DISCRETE ladder -- Var/Mean = 1.00, 0.45, 0.29, 0.21, 0.17 for
# D = 1, 3, 5, 7, 9 at mu = 136 -- and 57% of routes have empirical dispersion in the gap
# between the D=3 and D=1 rungs, while ~45 routes are overdispersed and cannot be
# represented at all. A NegBin parent has dispersion 1/p, so median-of-D NB reaches
# f(D)/p for continuous p: it fills the ladder and covers dispersion > 1.
#
# Augmentation is the one the genes model uses (models/genes/genes_final.jl):
#   sum of the D latent NB draws  ->  CRT  ->  Beta/Gamma conjugacy for p_k and r_k.
#
# CAVEAT worth checking empirically: dispersion is now f(D_k)/p_k, so D_k and p_k are only
# jointly identified through it. D_k may be less sharply determined than in the Poisson
# model, where D is the only dispersion knob. See report_D_identifiability below.

include("flights.jl")
include("../../helper/NegBinPMF.jl")   # logpmfOrderStatNegBin; not pulled in by flights.jl

struct flights_mednb <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    Dmax::Int64
    a::Float64        # shape of the r_k population prior (fixed, as in flights_hier)
    a0::Float64       # hyperprior shape for the population rate b
    b0::Float64       # hyperprior rate  for the population rate b
    alpha::Float64    # Beta prior on rho (the OddBinomial success prob)
    beta::Float64
    alpha_p::Float64  # Beta prior on the per-route p_k
    beta_p::Float64
end

# Chinese-restaurant-table samplers, as in models/genes/genes_final.jl
function sampleCRT_nb(Y, R)
    Y == 0 && return 0
    Y == 1 && return 1
    return 1 + sum(rand.(Bernoulli.([R / (R + i - 1) for i in 2:Y])))
end
function sampleCRTlecam_nb(Y, R, tol = 0.4)
    Ymax = R * (1 / tol - 1)
    (Y <= Ymax || Y <= 100) && return sampleCRT_nb(Y, R)
    out = sampleCRT_nb(Int(floor(Ymax)), R)
    mu = R * (SpecialFunctions.polygamma(0, R + Y) - SpecialFunctions.polygamma(0, R + Ymax))
    return out + rand(Poisson(mu))
end

function evalulateLogLikelihood(model::flights_mednb, state, data, info, row, col)
    Y = data["Y_NM"][row, col]
    k = data["routes_N"][row]
    r = state["r_R"][k]; p = state["p_R"][k]; D = state["D_R"][k]
    return D == 1 ? logpdf(NegativeBinomial(r, p), Y) :
                    logpmfOrderStatNegBin(Y, r, p, D, div(D, 2) + 1)
end

function sample_likelihood(model::flights_mednb, r, p, D, n = 1)
    d = D == 1 ? NegativeBinomial(r, p) : OrderStatistic(NegativeBinomial(r, p), D, div(D, 2) + 1)
    return n == 1 ? rand(d) : rand(d, n)
end

function sample_prior(model::flights_mednb, info = nothing, constantint = nothing)
    bmu = rand(Gamma(model.a0, 1 / model.b0))
    r_R = rand(Gamma(model.a, 1 / bmu), model.R)
    p_R = rand(Beta(model.alpha_p, model.beta_p), model.R)
    rho = 0.5
    @assert mod(model.Dmax, 2) == 1
    D_R = 2 * rand(Binomial((model.Dmax - 1) / 2, rho), model.R) .+ 1
    return Dict("r_R" => r_R, "p_R" => p_R, "D_R" => D_R, "p" => rho, "bmu" => bmu)
end

function forward_sample(model::flights_mednb; state = nothing, info = nothing)
    isnothing(state) && (state = sample_prior(model, info))
    routes_N = state["routes_N"]
    Y_NM = zeros(Int, model.N, model.M)
    for n in 1:model.N
        k = routes_N[n]
        Y_NM[n, 1] = sample_likelihood(model, state["r_R"][k], state["p_R"][k], state["D_R"][k])
    end
    return Dict("Y_NM" => Y_NM), state
end

function backward_sample(model::flights_mednb, data, state, mask = nothing;
                         skipupdatealways = nothing, skipupdate = nothing)
    Y_NM = copy(data["Y_NM"])
    routes_N = data["routes_N"]
    route_idx = data["route_idx"]::Vector{Vector{Int}}
    route_n = data["route_n"]::Vector{Int}
    r_R = copy(state["r_R"]); p_R = copy(state["p_R"])
    D_R = Int.(copy(state["D_R"])); rho = copy(state["p"]); bmu = copy(state["bmu"])
    @assert model.M == 1

    # ---- latent NB sums (Z2) and their CRT counts (Z1), accumulated per route ----
    nt = Threads.nthreads()
    Z2_nt = [zeros(Int, model.R) for _ in 1:nt]
    Z1_nt = [zeros(Int, model.R) for _ in 1:nt]
    @views @threads :static for n in 1:model.N
        tid = Threads.threadid()
        k = routes_N[n]
        r = r_R[k]; p = p_R[k]; D = D_R[k]
        if !isnothing(mask) && mask[n, 1] == 1
            Y_NM[n, 1] = sample_likelihood(model, r, p, D)
        end
        z2 = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, div(D, 2) + 1, NegativeBinomial(r, p))
        Z2_nt[tid][k] += z2
        z2 > 0 && (Z1_nt[tid][k] += sampleCRTlecam_nb(z2, D * r))
    end
    Z2_R = sum(Z2_nt); Z1_R = sum(Z1_nt)

    # ---- p_k | . : Beta conjugacy. Each flight contributes D_k draws from NB(r_k, p_k),
    #      and the sum of D_k*n_k iid NB(r_k, p) is NB(D_k*n_k*r_k, p).
    @views for k in 1:model.R
        p_R[k] = rand(Beta(model.alpha_p + D_R[k] * route_n[k] * r_R[k],
                           model.beta_p + Z2_R[k]))
    end

    # ---- r_k | . : Gamma conjugacy through the CRT counts ----
    @views for k in 1:model.R
        r_R[k] = rand(Gamma(model.a + Z1_R[k],
                            1 / (bmu + D_R[k] * route_n[k] * log(1 / p_R[k]))))
    end

    # ---- population rate b | r : conjugate (same hierarchy as flights_hier) ----
    bmu = rand(Gamma(model.a0 + model.R * model.a, 1 / (model.b0 + sum(r_R))))

    # ---- D_k update ----
    go = true
    if !isnothing(skipupdatealways) && "D_R" in skipupdatealways; go = false; end
    if !isnothing(skipupdate) && "D_R" in skipupdate; go = false; end

    if go
        logprobs_prior = [
            logbinomial(Int((model.Dmax - 1) ÷ 2), Int((d - 1) ÷ 2)) +
            (d - 1) * log(rho) / 2 + (model.Dmax - d) * log(1 - rho) / 2
            for d in 1:2:model.Dmax
        ]
        tabs_y = isnothing(mask) ?
            get!(() -> route_ycounts(Y_NM, route_idx), data, "route_ycounts") :
            route_ycounts(Y_NM, route_idx)
        uY = tabs_y[1]::Vector{Vector{Int}}
        cY = tabs_y[2]::Vector{Vector{Int}}
        cand = collect(1:2:model.Dmax); ncand = length(cand)
        tabs = [orderstat_grid_coefs(d, (d ÷ 2) + 1) for d in cand]

        @views @threads :static for k in 1:model.R
            r = r_R[k]; p = p_R[k]
            dist = NegativeBinomial(r, p)
            logprobs = copy(logprobs_prior)
            pA = Vector{Float64}(undef, model.Dmax + 1)
            pE = Vector{Float64}(undef, model.Dmax + 1)
            pB = Vector{Float64}(undef, model.Dmax + 1)
            uYk = uY[k]; cYk = cY[k]
            for i in eachindex(uYk)
                y = uYk[i]; c = cYk[i]
                # parent cdf/pmf do not depend on the candidate D (grid pmf is
                # parent-agnostic: it takes F and f as numbers)
                F = cdf(dist, y); f = pdf(dist, y); lf = logpdf(dist, y)
                for ci in 1:ncand
                    d = cand[ci]
                    if d == 1
                        logprobs[ci] += c * lf
                    else
                        j = (d ÷ 2) + 1
                        v = logpmf_orderstat_grid(F, f, d, j, tabs[ci], pA, pE, pB)
                        isnan(v) && (v = logpmfOrderStatNegBin(y, r, p, d, j))
                        logprobs[ci] += c * v
                    end
                end
            end
            D_R[k] = 2 * argmax(rand(Gumbel(0, 1), ncand) .+ logprobs) - 1
        end
        rho = rand(Beta(model.alpha + (sum(D_R) - model.R) / 2,
                        model.beta + (model.Dmax * model.R - sum(D_R)) / 2))
    end

    return data, Dict("r_R" => r_R, "p_R" => p_R, "D_R" => D_R, "p" => rho, "bmu" => bmu)
end
