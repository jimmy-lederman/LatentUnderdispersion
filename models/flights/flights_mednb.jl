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
    Y <= 0 && return 0
    Y == 1 && return 1
    # Allocation-free. The idiomatic form, 1 + sum(rand.(Bernoulli.([R/(R+i-1) for i in 2:Y]))),
    # builds three O(Y) temporaries per call -- the probability vector, the vector of
    # Bernoulli objects, and the vector of draws. Here Y is the latent NB sum, roughly
    # D * mean ~ 760 for the flights data, and the call happens once per flight per sweep
    # (42,773 times), so that is ~32M draws behind ~128k array allocations every sweep.
    # Same draws, same distribution: rand() < R/(R+i-1) is rearranged to avoid the divide.
    c = 1
    @inbounds for i in 2:Y
        rand() * (R + i - 1) < R && (c += 1)
    end
    return c
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
    rho = rand(Beta(model.alpha, model.beta))   #drawn from its prior (was hardcoded 0.5)
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

# Mean-preserving griddy update of p_k.
#
# The conjugate step cannot learn p at this data scale: p | . ~ Beta(a + D*n*r, b + Z2)
# with Z2 the latent sum drawn UNDER the current p, so the conditional is centred on where
# it already is (measured: from starts 0.15/0.30/0.60/0.90 it moved to 0.143/0.282/0.587/
# 0.898 in 300 sweeps). The informative direction changes dispersion at FIXED mean, and
# that direction moves p and r together via r = mu*p/(1-p) -- a coordinate move cannot
# follow it.
#
# So reparameterise (r, p) -> (mu, p) and update p | mu. The Jacobian of r = mu*p/(1-p) at
# fixed mu is dr/dmu = p/(1-p), giving the log-weight below. Scoring is on the OBSERVED Y
# through the order-statistic pmf, not on latent draws generated under the current p.
# Verified beforehand that a route's own flights do pin its dispersion this way: implied
# dispersion at the profile maximum matched the empirical to a few percent, with 1.6-11.5
# nats of curvature at +/-0.1 in p.
const PGRID = collect(0.03:0.04:0.95)

# How often the marginal p move runs, and how many proposals it makes when it does.
# Between those sweeps p uses its conjugate update. Running it periodically rather than
# only during burn-in matters: the conjugate step is near-self-confirming, so a burn-in-only
# schedule would leave p frozen wherever the marginal step last put it, giving an
# artificially narrow posterior for the dispersion.
const P_MH_EVERY = 10
const P_MH_NPROP = 5
const P_MH_SD    = 0.6      # random-walk sd on the logit scale

# marginal log-target for p at fixed mean mu, on the curve r = mu*p/(1-p).
# Same target the griddy scores; see griddy_p! for why it is the observed-Y marginal.
@inline function logtarget_p(pc, mu, D, j, uYk, cYk, model, T, pA, pE, pB, bmu)
    (pc <= 0 || pc >= 1) && return -Inf
    rc = mu * pc / (1 - pc)
    (rc <= 0 || !isfinite(rc)) && return -Inf
    dist = NegativeBinomial(rc, pc)
    acc = 0.0
    @inbounds for i in eachindex(uYk)
        y = uYk[i]
        F = cdf(dist, y); f = pdf(dist, y); lf = logpdf(dist, y)
        v = D == 1 ? lf : logpmf_orderstat_grid(F, f, D, j, T, pA, pE, pB, lf)
        isnan(v) && (v = -Inf)
        acc += cYk[i] * v
    end
    acc += (model.a - 1) * log(rc) - bmu * rc                       # prior on r
    acc += (model.alpha_p - 1) * log(pc) + (model.beta_p - 1) * log(1 - pc)
    acc += log(pc) - log(1 - pc)                                    # Jacobian dr/dmu
    return acc
end

# Random-walk Metropolis on logit(p), holding the fitted mean fixed. One likelihood
# evaluation per proposal instead of the grid's 24, and p stays continuous rather than
# being pinned to lattice points. Proposal is symmetric in logit space, so the accept
# ratio carries the log|dp/dtheta| = log(p(1-p)) terms and nothing else.
function mh_p!(r_R, p_R, D_R, uY, cY, model, tabs_by_D, bmu)
    pA = Vector{Float64}(undef, model.Dmax + 1)
    pE = Vector{Float64}(undef, model.Dmax + 1)
    pB = Vector{Float64}(undef, model.Dmax + 1)
    nacc = 0; ntry = 0
    @inbounds for k in 1:model.R
        uYk = uY[k]; isempty(uYk) && continue
        cYk = cY[k]; D = D_R[k]; j = div(D, 2) + 1; T = tabs_by_D[D]
        mu = r_R[k] * (1 - p_R[k]) / p_R[k]
        (mu <= 0 || !isfinite(mu)) && continue
        pcur = p_R[k]
        lcur = logtarget_p(pcur, mu, D, j, uYk, cYk, model, T, pA, pE, pB, bmu) +
               log(pcur) + log(1 - pcur)
        for _ in 1:P_MH_NPROP
            th = log(pcur / (1 - pcur)) + P_MH_SD * randn()
            pp = 1 / (1 + exp(-th))
            lp = logtarget_p(pp, mu, D, j, uYk, cYk, model, T, pA, pE, pB, bmu) +
                 log(pp) + log(1 - pp)
            ntry += 1
            if log(rand()) < lp - lcur
                pcur = pp; lcur = lp; nacc += 1
            end
        end
        p_R[k] = pcur
        r_R[k] = mu * pcur / (1 - pcur)
    end
    return nacc / max(ntry, 1)
end

function griddy_p!(r_R, p_R, D_R, uY, cY, model, tabs_by_D, bmu)
    R = model.R
    pA = Vector{Float64}(undef, model.Dmax + 1)
    pE = Vector{Float64}(undef, model.Dmax + 1)
    pB = Vector{Float64}(undef, model.Dmax + 1)
    logw = Vector{Float64}(undef, length(PGRID))
    @inbounds for k in 1:R
        uYk = uY[k]; isempty(uYk) && continue
        cYk = cY[k]
        D = D_R[k]; j = div(D, 2) + 1
        mu = r_R[k] * (1 - p_R[k]) / p_R[k]      # hold the fitted mean fixed
        (mu <= 0 || !isfinite(mu)) && continue
        T = tabs_by_D[D]
        for (gi, pc) in enumerate(PGRID)
            rc = mu * pc / (1 - pc)
            dist = NegativeBinomial(rc, pc)
            acc = 0.0
            for i in eachindex(uYk)
                y = uYk[i]
                F = cdf(dist, y); f = pdf(dist, y); lf = logpdf(dist, y)
                v = D == 1 ? lf : logpmf_orderstat_grid(F, f, D, j, T, pA, pE, pB, lf)
                isnan(v) && (v = -Inf)
                acc += cYk[i] * v
            end
            # prior on r at the transformed point, prior on p, and the Jacobian p/(1-p)
            acc += (model.a - 1) * log(rc) - bmu * rc
            acc += (model.alpha_p - 1) * log(pc) + (model.beta_p - 1) * log(1 - pc)
            acc += log(pc) - log(1 - pc)
            logw[gi] = acc
        end
        gbest = 1; best = -Inf
        for gi in eachindex(PGRID)
            v = logw[gi] + rand(Gumbel(0, 1))
            v > best && (best = v; gbest = gi)
        end
        p_R[k] = PGRID[gbest]
        r_R[k] = mu * p_R[k] / (1 - p_R[k])
    end
    return nothing
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

    # distinct Y per route, needed by BOTH the p griddy step and the D update
    tabs_y = isnothing(mask) ?
        get!(() -> route_ycounts(Y_NM, route_idx), data, "route_ycounts") :
        route_ycounts(Y_NM, route_idx)
    uY = tabs_y[1]::Vector{Vector{Int}}
    cY = tabs_y[2]::Vector{Vector{Int}}
    tabs_by_D = Dict(d => orderstat_grid_coefs(d, (d ÷ 2) + 1) for d in 1:2:model.Dmax)

    # ---- p_k | . : Beta conjugacy. Each flight contributes D_k draws from NB(r_k, p_k),
    #      and the sum of D_k*n_k iid NB(r_k, p) is NB(D_k*n_k*r_k, p).
    # The clamp is a numerical guard, not a modelling choice. Dispersion is f(D)/p, so
    # p -> 0 means an ever more overdispersed parent; the latent sums Z2 then grow without
    # bound and OVERFLOW Int64, which surfaces as a negative Beta parameter and kills the
    # chain (observed at D = 9). At PMIN = 1e-4 the implied dispersion ceiling is
    # f(D)/PMIN >= 1670, against a maximum empirical route dispersion of about 2 -- so the
    # bound is far outside the region the data support and never binds in practice.
    # p update: the MARGINAL move every P_MH_EVERY sweeps, the conjugate one otherwise.
    # Never both in the same sweep -- the conjugate draw would just be overwritten, and it
    # is the marginal move that actually learns p (the conjugate conditional is centred on
    # wherever p already is; measured, it does not move at this data scale).
    sweep = get(state, "sweep", 0) + 1
    do_marginal = (sweep % P_MH_EVERY == 0)
    if !do_marginal
        @views for k in 1:model.R
            p_R[k] = clamp(rand(Beta(model.alpha_p + D_R[k] * route_n[k] * r_R[k],
                                     model.beta_p + Z2_R[k])), 1e-4, 1 - 1e-12)
        end
    end

    # ---- r_k | . : Gamma conjugacy through the CRT counts ----
    @views for k in 1:model.R
        r_R[k] = rand(Gamma(model.a + Z1_R[k],
                            1 / (bmu + D_R[k] * route_n[k] * log(1 / p_R[k]))))
    end

    # ---- p_k | mu : mean-preserving griddy move (replaces the conjugate p step's role
    #      as the dispersion update; see griddy_p! above for why the conjugate one cannot
    #      learn p at this data scale). Resets r_k along the curve r = mu*p/(1-p).
    accrate = do_marginal ? mh_p!(r_R, p_R, D_R, uY, cY, model, tabs_by_D, bmu) : NaN

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
                        v = logpmf_orderstat_grid(F, f, d, j, tabs[ci], pA, pE, pB, lf)
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

    return data, Dict("r_R" => r_R, "p_R" => p_R, "D_R" => D_R, "p" => rho, "bmu" => bmu,
                      "sweep" => sweep, "p_accrate" => accrate)
end
