include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
include("../genes/genes_polya/PolyaGammaHybridSamplers.jl/src/pghybrid.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
using SpecialFunctions
using Base.Threads

# The sampler calls BLAS from INSIDE @threads loops (the tau precompute over m and the beta
# loop over n), and OpenBLAS sizes its pool from the machine's physical core count rather
# than from the cgroup allocation. Left alone that gives nthreads x ncores threads: measured
# 0.67 -> 0.58 s/sweep here just from pinning it. On a cluster node it is far worse -- a
# 2-CPU job on a 64-core node spawns 64 BLAS threads fighting over 2 cores, which is the
# failure mode that previously turned 31-second cells into hours.
#
# Nothing here wants a BLAS thread pool anyway: the products inside the threaded loops are
# Q x Q with Q = 25, and the largest product outside any threaded loop (Beta_NQ * Tau_QM) is
# ~23 MFlops. Also set OPENBLAS_NUM_THREADS=1 in job scripts, since this only takes effect
# once the module is loaded.
BLAS.set_num_threads(1)

function sampleCRT(Y, R)
    Y <= 1 && return Y
    out = 1
    @inbounds for i in 2:Y
        out += rand() < R / (R + i - 1)
    end
    return out
end

# CRT with a le Cam tail approximation, as used by the genes model. sampleCRT is O(Y)
# with a Bernoulli draw per unit, and the counts it is applied to here are aggregated
# over all 3104 counties, so on cumulative COVID counts Y routinely reaches 1e6+ and the
# exact loop alone would dominate the sweep. Beyond Y_max = R(1/tol - 1) the remaining
# Bernoullis have success probability below `tol` and are essentially independent, so
# their sum is Poisson to that order; the mean R*(psi(R+Y) - psi(R+Y_max)) is exact.
# Tolerance of the le Cam tail approximation: tables with Bernoulli probability
# R/(R+i-1) >= tol are drawn exactly, the rest are replaced by a Poisson with the same
# mean. That Poisson has variance sum p_i instead of sum p_i(1-p_i), so the tail's
# variance is inflated by up to a factor 1/(1-tol). The paper's value 0.4 is a visible
# bias once the theta step mixes (the Schein test only hides it under the slow thinning
# sampler: hybrid sampler, 1200 x 30 sweeps, phi mean -2.6% at p = 3e-4 with 0.4, passes
# with an exact CRT). 0.02 makes the head exact up to i = 49R and bounds the tail's
# variance error by 2%; on the full matrix (R ~ 1e3, Y ~ 1e6) that is under 1% of the
# CRT count's variance at ~0.5 s/sweep. <= 0 means exact everywhere (O(Y) per draw).
const CRT_TOL = Ref(0.02)
# Test switch: draw tau columns with an explicit precision matrix and MvNormal (the
# original serial formulation) instead of the Cholesky-factor draw, to bisect the tau
# bias seen in the realistic-regime Schein test.
const TAU_NAIVE = Ref(false)
# Group moves on each factor's trajectory (phi_group_move!), off by default; the cluster
# driver and the test harness switch it on. See the function for what it does.
const PHI_GROUP = Ref(false)
const PHI_GROUP_J = Ref(8)      # number of smooth basis functions (cosines of order 1..J)
function sampleCRTlecam(Y::Integer, R::Real, tol=CRT_TOL[])
    tol <= 0 && return sampleCRT(Y, R)
    Y_max = floor(Int, R * (1/tol - 1))
    if Y <= Y_max || Y <= 100
        return sampleCRT(Y, R)
    else
        out = sampleCRT(Y_max, R)
        mu = R * (polygamma(0, R + Y) - polygamma(0, R + Y_max))
        return out + rand(Poisson(mu))
    end
end


struct covid4 <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    Q::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    g::Float64 
    h::Float64
    start_V1::Float64
    start_V2::Float64
    alpha0::Float64
    beta0::Float64
    tauc::Float64
    taud::Float64
    start_tau::Float64
    # When true, the rate of the county-factor prior theta_{i,k} ~ Gamma(a, b) is learned
    # rather than held at `b`, pooling across all N counties:
    #     b | theta ~ Gamma(e0 + N*K*a,  f0 + sum(theta))
    # This is the Section 7.2 analogue of the hierarchical mu_k added to the flights model
    # for referee comment 2(a), and uses the same Gamma(1,1) hyperprior as Section 6.3
    # (MatrixMF's sample_rate_hyper / sample_rate_prior). `b` is then a state variable, and
    # the flat model is recovered exactly by hier_b = false.
    hier_b::Bool
    # Interleave the multiplicative scale move on (theta_{:,k}, phi_{k,:}). Pure sampler
    # change -- the target posterior is untouched. See scale_move! below.
    scale_move::Bool
    # Interleave a marginal Metropolis step on eps. Also a pure sampler change. See
    # eps_marginal_move! below.
    eps_mh::Bool
    # When false, drop Y_{i,t-1} from the rate: mu = eps + sum_k theta phi. This turns the
    # model of CUMULATIVE counts into a model of INCREMENTS, and exists to test how much of
    # the factors' poor identifiability is caused by the offset. The observation noise of a
    # median-of-D Poisson scales with sqrt(mu), so with the offset mu ~ 1200 and the growth
    # signal ~13 give a per-cell signal-to-noise near 1; without it mu ~ 13 and the same
    # signal gives ~8. Feed increments as the data when this is false.
    use_offset::Bool
    # Interleave the pairwise SHEAR move on (theta_{:,k}, phi_{j,:}). With the scale move
    # this spans the full 2x2 mixing group under which the factorisation is non-identified.
    # Pure sampler change. See shear_move! below.
    shear_move::Bool
    # Collapse the multinomial thinning: theta, phi and eps are updated by univariate slice
    # sampling on the exact Poisson-with-offset conditionals given the latent sums
    # S_{i,t} | Y, D (which stay augmented). The thinning is what makes the augmented
    # sampler crawl: given the split of S ~ 1e5 into offset and factor counts, theta is
    # known to sd 1/sqrt(D*g) while the data pin it only to ~ sqrt(mu/D)/g, so the
    # fraction of missing information is ~ 1 - g/mu and the DA autocorrelation time is
    # ~ mu/g sweeps (10^2 - 10^4 on the COVID matrix). See collapsed_update! below.
    collapsed::Bool
    # Replace the gamma-Markov chain on phi with iid phi_{k,t} ~ Gamma(start_V1, start_V2):
    # a "regular MF" prior, to test whether the temporal prior is what blocks mixing. Test
    # flag only; the scale / shear moves and logprior_phi_* assume the chain prior, so run
    # with scale_move = false.
    iid_phi::Bool
    # With collapsed = true: keep the multinomial thinning, the conjugate eps draw and the
    # CRT backward / Gamma forward pass for phi (the PGDS time-factor augmentation of the
    # paper), and collapse ONLY the theta draw. Order within the sweep is z -> eps -> phi ->
    # theta so the collapsed step comes last and z is then discarded (a partially collapsed
    # Gibbs sampler, valid in this order and not in others).
    phi_da::Bool
    # theta_{.,k} ~ Dirichlet(a, ..., a) on the county simplex instead of iid Gamma(a, b):
    # the interpreted "share of factor k's activity in county i" IS the parameter, the
    # per-factor scale lives in phi_{k,t} ("cases attributable to factor k on day t"), and
    # the theta/phi scale non-identifiability is gone by construction (b, the scale move
    # and the s_k normalisation become moot). Marginally the Gamma model already had
    # theta_k / s_k ~ Dirichlet(a); what changes is that the gamma-Markov prior now sits
    # on the scaled trajectory, so c must be set for that scale (see c_for_scale).
    # Requires collapsed = true (the theta step is a pairwise-exchange slice move on the
    # simplex, theta_simplex!); phi keeps the PGDS pass when phi_da = true.
    dir_theta::Bool
end
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co, false, false, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co, ip) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co, ip, false, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co, ip, pd) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, co, ip, pd, false)

# The gamma-Markov chain phi_t | phi_{t-1} ~ Gamma(c phi_{t-1} + d, c) has relative step sd
# 1/sqrt(c phi): its smoothness depends on the scale of phi. c_for_scale(r, phibar) is
# the c that gives relative sd r at a typical factor size phibar. The paper's c = 100 at
# phi ~ 25 is r ~ 2%; with Dirichlet theta, phi carries the national-scale counts, so the
# same smoothness needs c ~ 1/(0.02^2 phibar).
c_for_scale(r, phibar) = 1 / (r^2 * phibar)
# flat-prior constructors, so the 18-, 19- and 20-argument calls elsewhere still work
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, sh, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, false, false, false, true, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, false, false, true, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, false, true, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, true, false)
covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo) =
    covid4(N, M, K, Q, Dmax, a, b, c, d, g, h, sV1, sV2, al, be, tc, td, st, hb, sm, em, uo, false)


function evalulateLogLikelihood(model::covid4, state, data, info, row, col)
    @assert !isnothing(info)
    Ylast = !model.use_offset ? 0 : (col == 1 ? info["Y0_N"][row] : data["Y_NM"][row,col-1])
    Y = data["Y_NM"][row,col]
    D = state["D_NM"][row,col]
    eps = state["eps"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    mu = sum(U_NK[row,:] .* V_KM[:,col])
    rate = Ylast+mu+eps
    if D == 1
        return logpdf(Poisson(rate), Y)
    else
        return logpmfOrderStatPoisson(Y,rate,D,div(D,2)+1)
    end
end

function sample_likelihood(model::covid4, mu,D,n=1)
    if D == 1
        return rand(Poisson(mu),n)
    else
        return rand(OrderStatistic(Poisson(mu),D, div(D,2)+1),n)
    end
end

function sample_prior(model::covid4,info=nothing,constantinit=nothing)
    pass = false
    # b is drawn from its own prior when it is free, so chains start dispersed in the
    # county-factor scale too rather than all at the same b.
    b_mu = model.hier_b ? sample_rate_prior() : model.b
    U_NK = rand(Gamma(model.a, 1/b_mu), model.N, model.K)
    if model.dir_theta      # columns on the simplex: normalised Gamma(a, 1) draws
        U_NK = rand(Gamma(model.a, 1.0), model.N, model.K)
        U_NK ./= sum(U_NK, dims=1)
    end
    #V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1 || model.iid_phi
                    V_KM[k,m] = rand(Gamma(model.start_V1,1/model.start_V2))
                    #V_KM[k,m] = rand(Gamma(model.c*model.start_V+model.d,1/model.c))
                else
                    V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1]+model.d,1/model.c))
                end
            end
        end
    end
    eps = rand(Gamma(model.g, 1/model.h))

    #sigma_N = rand(InverseGamma(model.alpha0, model.beta0), model.N)
    #sigma_M = rand(InverseGamma(model.alpha0, model.beta0), model.M)
    sigma2_beta = rand(InverseGamma(model.alpha0, model.beta0))
    sigma2_tau = rand(InverseGamma(model.alpha0, model.beta0))
    # sigma2_beta = 1
    # sigma2_tau = 1

    Beta_NQ = rand(Normal(0,sqrt(sigma2_beta)),model.N,model.Q)
    #Beta_NQ = rand(Normal(0,.01),model.N,model.Q)
    Tau_QM = zeros(model.Q, model.M)
    @views for m in 1:model.M
        @views for q in 1:model.Q
            if m == 1
                Tau_QM[q,m] = rand(Normal(model.tauc*model.start_tau+model.taud,sqrt(sigma2_tau)))
                #Tau_QM[q,m] = rand(Normal(model.tauc*model.start_tau+model.taud,.01))
            else
                Tau_QM[q,m] = rand(Normal(model.tauc*Tau_QM[q,m-1]+model.taud,sqrt(sigma2_tau)))
                #Tau_QM[q,m] = rand(Normal(model.tauc*model.start_tau+model.taud,.01))
            end
        end
    end

    D_NM = 2*rand.(Binomial.(Int((model.Dmax - 1)/2), logistic.(Beta_NQ * Tau_QM))) .+ 1


    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, #"R_KTS" => R_KTS,
    "eps"=>eps, "D_NM"=>D_NM, "b_mu"=>b_mu,
    "sigma2_beta"=>sigma2_beta, "sigma2_tau"=>sigma2_tau, "Beta_NQ"=>Beta_NQ, "Tau_QM"=>Tau_QM,
     "Y0_N"=>info["Y0_N"])
    return state
end

function forward_sample(model::covid4; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    eps = state["eps"]
    Y0_N = state["Y0_N"]

    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    D_NM = state["D_NM"]

    Y_NM = zeros(Int, model.N, model.M)

    for m in 1:model.M
        for n in 1:model.N
            D = D_NM[n,m]
            if m == 1
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y0_N[n] + eps + sum(U_NK[n,:] .* V_KM[:,m])), D, div(D,2)+1))
            else
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Y_NM[n,m-1] + eps + sum(U_NK[n,:] .* V_KM[:,m])),D, div(D,2)+1))
            end
        end 
    end

    data = Dict("Y_NM" => Y_NM)

    return data, state 
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

# log prior of one row of phi under the gamma-Markov chain of Eq. (14):
#   phi_{k,1} ~ Gamma(start_V1, rate start_V2)
#   phi_{k,t} ~ Gamma(c*phi_{k,t-1} + d, rate c)
# Evaluated at phi/kappa without materialising the scaled row.
function logprior_phi_row(model::covid4, V_KM, k, kappa)
    @inbounds begin
        f1 = V_KM[k,1] / kappa
        lp = (model.start_V1 - 1)*log(f1) - model.start_V2*f1 -
             SpecialFunctions.loggamma(model.start_V1) + model.start_V1*log(model.start_V2)
        prev = f1
        for t in 2:model.M
            ft = V_KM[k,t] / kappa
            al = model.c*prev + model.d
            lp += al*log(model.c) - SpecialFunctions.loggamma(al) +
                  (al - 1)*log(ft) - model.c*ft
            prev = ft
        end
    end
    return lp
end

# Interleaved multiplicative scale move.
#
# The likelihood depends on theta and phi only through the products theta_{i,k} phi_{k,t},
# so for any kappa > 0
#     theta_{:,k} -> kappa * theta_{:,k},   phi_{k,:} -> phi_{k,:} / kappa
# leaves every mu_{i,t} exactly unchanged. The posterior therefore has a ridge in this
# direction that only the two priors pin down, and single-site Gibbs can only crawl along
# it: theta's update takes phi's scale as given and vice versa, so neither ever moves the
# ridge coordinate by more than the noise in the other block.
#
# The COMPLETE-DATA likelihood is invariant too, which is what makes the move cheap. Its
# two terms are
#     sum_{i,t} D_{i,t} theta_{i,k} phi_{k,t}                        (kappa cancels)
#     sum_i Y_NK[i,k] log theta_{i,k} + sum_t Y_MK[t,k] log phi_{k,t}
# and the second changes by (sum_i Y_NK[i,k] - sum_t Y_MK[t,k]) log kappa, which is zero
# because both sums are the total count allocated to factor k. So the Metropolis ratio is
# the prior ratio times the Jacobian kappa^(N-M), with no reference to the data at all.
#
# The proposal is a random walk on log kappa with sd matching the ridge's own width: with
# shape a the ridge coordinate has posterior relative sd of order 1/sqrt(N*a), so steps
# much larger than that are rejected and much smaller ones waste the move.
#
# This does not change the target distribution. It is the same model and the same
# posterior; the chain just jumps along the ridge instead of crawling.
function scale_move!(model::covid4, U_NK, V_KM, b_mu, nattempt::Int)
    nacc = 0
    sd = 1 / sqrt(model.N * model.a)
    @inbounds for k in 1:model.K
        S_k = 0.0
        for i in 1:model.N; S_k += U_NK[i,k]; end
        lp_cur = logprior_phi_row(model, V_KM, k, 1.0)
        for _ in 1:nattempt
            lkap = sd * randn()
            kap = exp(lkap)
            lp_new = logprior_phi_row(model, V_KM, k, kap)
            # theta prior ratio (closed form) + phi prior ratio + Jacobian
            logalpha = (model.a - 1)*model.N*lkap - b_mu*(kap - 1)*S_k +
                       (lp_new - lp_cur) + (model.N - model.M)*lkap
            if log(rand()) < logalpha
                for i in 1:model.N; U_NK[i,k] *= kap; end
                for t in 1:model.M; V_KM[k,t] /= kap; end
                S_k *= kap
                lp_cur = lp_new
                nacc += 1
            end
        end
    end
    return nacc / (model.K * nattempt)
end

# Scratch space for the D-update, allocated once per thread-chunk rather than per cell.
# `tabs` holds the restricted-multinomial coefficient table for each candidate order,
# `logprior` the order-independent part of the OddBinomial prior, and pA/pE/pB are the
# power buffers logpmf_orderstat_grid writes into.
struct DUpdateWork
    cand::Vector{Int}
    jj::Vector{Int}
    tabs::Vector{Matrix{Float64}}
    logbin::Vector{Float64}
    logprobs::Vector{Float64}
    pA::Vector{Float64}
    pE::Vector{Float64}
    pB::Vector{Float64}
end

function DUpdateWork(model::covid4)
    cand = collect(1:2:model.Dmax)
    jj = [d ÷ 2 + 1 for d in cand]
    half = (model.Dmax - 1) ÷ 2
    DUpdateWork(cand, jj,
        [orderstat_grid_coefs(cand[i], jj[i]) for i in eachindex(cand)],
        [logbinomial(half, (d - 1) ÷ 2) for d in cand],
        Vector{Float64}(undef, length(cand)),
        Vector{Float64}(undef, model.Dmax + 1),
        Vector{Float64}(undef, model.Dmax + 1),
        Vector{Float64}(undef, model.Dmax + 1))
end

# Draw D_{n,m} | Y, mu, rho from the OddBinomial prior times the order-statistic pmf.
#
# The parent cdf/pdf/logpdf at Y do not depend on the candidate order, so they are
# evaluated ONCE per cell here instead of once per candidate inside
# logpmfOrderStatPoisson (which also takes a lock on the shared coefficient-table cache
# on every call -- a serialization point that would have capped threading). The pmf
# itself goes through the same cancellation-free grid expansion as before, so the
# log-probabilities are the values the previous code computed.
function update_D(model::covid4, Y, mu, p, w::DUpdateWork)
    logp   = log(p)
    log1mp = log1p(-p)
    dist   = Poisson(mu)
    # Parent quantities from the Poisson pmf recurrence (helper/OrderStatsSampling.jl,
    # os_consts(::Poisson)): no incomplete gamma per cell, and more accurate near Y ~ mu.
    # The generic path is used automatically where the recurrence would underflow.
    C  = os_consts(dist, Y)
    F  = C.FY
    f  = C.pdfY
    lf = C.lpY

    half = (model.Dmax - 1) ÷ 2
    @inbounds for i in eachindex(w.cand)
        d = w.cand[i]
        lpmf = if d == 1
            lf
        else
            v = logpmf_orderstat_grid(F, f, d, w.jj[i], w.tabs[i], w.pA, w.pE, w.pB)
            isnan(v) ? logpmfOrderStatPoisson(Y, mu, d, w.jj[i], compute=false) : v
        end
        # OddBinomial prior on D = 2k+1 with k ~ Binomial(half, p): C(half,k) p^k (1-p)^(half-k).
        # The zero exponents must be skipped rather than multiplied through. p_{n,m} is
        # logistic(beta_n' tau_m) and does saturate to exactly 0 or 1 in Float64 for the
        # extreme cells, and there 0*log(0) = 0*(-Inf) is NaN, not 0. Writing the term as
        # (d-1)*log(p)/2 + (Dmax-d)*log(1-p)/2 therefore puts a NaN in the k=0 entry when
        # p=0 and in the k=half entry when p=1. argmax happens to propagate that NaN and so
        # returns the right candidate at p=0 and at p=1, but any code that compares
        # candidates with `>` -- as a scalar Gumbel loop does -- silently skips it and falls
        # through to D=1. Guarding the exponents keeps the term exact (p^0 = 1) and makes
        # the result independent of how the argmax is written.
        kk = (d - 1) >> 1
        tp   = kk > 0          ? kk * logp             : 0.0
        t1mp = (half - kk) > 0 ? (half - kk) * log1mp  : 0.0
        # Summed in the same association as the previous code -- (d-1)*logp/2 is bitwise
        # equal to kk*logp since the scaling is by a power of two -- so away from the
        # saturated corners the log-probabilities are unchanged bit for bit.
        w.logprobs[i] = lpmf + w.logbin[i] + tp + t1mp
    end

    # Gumbel-max over the candidates. Scalar draws in order are bitwise identical to
    # rand(Gumbel(0,1), n), and `>` reproduces argmax's first-wins tie-breaking.
    best_val = -Inf
    best_i = 1
    @inbounds for i in eachindex(w.cand)
        v = w.logprobs[i] + rand(Gumbel(0, 1))
        if v > best_val
            best_val = v
            best_i = i
        end
    end
    D = w.cand[best_i]
    @assert D >= 1 && D <= model.Dmax
    return D
end


# Total log-likelihood of the data under a given eps, with the LATENT COUNTS INTEGRATED OUT.
# Threaded over columns; uses the same cancellation-free grid pmf as the D update, so the
# cost is about one D-update candidate over the whole matrix.
function total_loglik_eps(model::covid4, Y_NM, Y0_N, mu_NM, D_NM, eps, works)
    nt = length(works)
    acc = zeros(Float64, nt)
    @views @threads :static for m in 1:model.M
        tid = Threads.threadid()
        w = works[tid]
        s = 0.0
        @inbounds for n in 1:model.N
            Ylast = !model.use_offset ? 0 : (m == 1 ? Y0_N[n] : Y_NM[n,m-1])
            mu = Ylast + eps + mu_NM[n,m]
            D = D_NM[n,m]
            Y = Y_NM[n,m]
            dist = Poisson(mu)
            if D == 1
                s += logpdf(dist, Y)
            else
                idx = (D + 1) >> 1
                j = w.jj[idx]
                v = logpmf_orderstat_grid((C_ = os_consts(dist, Y); C_.FY), C_.pdfY, D, j,
                                          w.tabs[idx], w.pA, w.pE, w.pB)
                isnan(v) && (v = logpmfOrderStatPoisson(Y, mu, D, j, compute=false))
                s += v
            end
        end
        acc[tid] += s
    end
    return sum(acc)
end

# Marginal Metropolis step on eps.
#
# eps is a single scalar added to every one of ~900k rates, so its AUGMENTED-data conditional
# Gamma(g + sum_noise, h + sum D) has relative width 1/sqrt(eps * sum D) -- about 0.15% on
# the full data. Its MARGINAL posterior is far wider than that, because eps trades off
# against the overall level of theta'phi. Conditioning on the latent counts therefore pins
# eps to whatever the current allocation implies, and the Gibbs chain crawls: measured on
# real data, eps drifts monotonically from 0.20 to 0.004 over 5000 sweeps without settling,
# and on synthetic data it simply stays wherever it is initialised.
#
# This step proposes eps on the log scale and accepts against the ACTUAL likelihood with the
# counts integrated out, so it moves by the marginal width instead. It targets
# pi(eps | theta, phi, D, Y), which is the correct conditional for eps in the parameter
# posterior, and it runs at the end of the sweep where the latent counts are no longer part
# of the state -- so this is ordinary Metropolis-within-Gibbs, not a partially-collapsed
# construction needing a special argument.
#
# Measured value (oracle experiment, synthetic): fixing eps is worth roughly 3x in run
# length -- eps at its true value with 9000 sweeps matches eps from the prior with 27000.
# Cost is one full-data likelihood evaluation per proposal.
function eps_marginal_move!(model::covid4, Y_NM, Y0_N, mu_NM, D_NM, eps, works,
                            nattempt::Int, sd::Float64)
    ll = total_loglik_eps(model, Y_NM, Y0_N, mu_NM, D_NM, eps, works)
    nacc = 0
    for _ in 1:nattempt
        lk = sd * randn()
        eps_new = eps * exp(lk)
        ll_new = total_loglik_eps(model, Y_NM, Y0_N, mu_NM, D_NM, eps_new, works)
        # Gamma(g, rate h) prior, plus the log-scale Jacobian
        lp_new = (model.g - 1)*log(eps_new) - model.h*eps_new
        lp_cur = (model.g - 1)*log(eps)     - model.h*eps
        if log(rand()) < (ll_new - ll) + (lp_new - lp_cur) + lk
            eps = eps_new; ll = ll_new; nacc += 1
        end
    end
    return eps, nacc / nattempt
end

# log prior of one row of phi given as a vector, under the gamma-Markov chain of Eq. (14).
# Returns -Inf if any entry is non-positive, so an infeasible proposal is rejected by the
# acceptance test rather than by a separate check.
function logprior_phi_vec(model::covid4, v::AbstractVector)
    @inbounds begin
        f1 = v[1]
        f1 > 0 || return -Inf
        lp = (model.start_V1 - 1)*log(f1) - model.start_V2*f1 -
             SpecialFunctions.loggamma(model.start_V1) + model.start_V1*log(model.start_V2)
        prev = f1
        for t in 2:model.M
            ft = v[t]
            ft > 0 || return -Inf
            al = model.c*prev + model.d
            lp += al*log(model.c) - SpecialFunctions.loggamma(al) + (al - 1)*log(ft) - model.c*ft
            prev = ft
        end
    end
    return lp
end

# Pairwise shear move on the factors.
#
# The likelihood sees theta and phi only through mu = sum_k theta_{:,k} phi_{k,:}'. For any
# pair (j, k) and scalar a, the SHEAR
#     theta_{:,k} <- theta_{:,k} + a*theta_{:,j},     phi_{j,:} <- phi_{j,:} - a*phi_{k,:}
# is [theta_j theta_k]*A and A^{-1}*[phi_j; phi_k] with A = [1 a; 0 1], so it leaves every
# mu_{i,t} EXACTLY unchanged. The scale move is the diagonal member of the same 2x2 group;
# this is the off-diagonal one, and together they span the transformations under which a
# non-negative factorisation is not identified -- which is where the chains disagree.
#
# Because mu is unchanged the MARGINAL likelihood is invariant and det A = 1, so the
# Metropolis ratio is the prior ratio alone: theta's Gamma(a, b) on the changed column k,
# and the gamma-Markov chain on the changed row j. Non-negativity is the feasible region;
# a is drawn uniformly on the feasible interval, which is symmetric (see the body), so
# there is no correction term.
#
# Unlike the scale move this is NOT invariant under the complete-data likelihood (the
# per-factor products change), so it must run at the END of the sweep, where the latent
# counts have been used and discarded and the state is just the parameters -- ordinary
# Metropolis-within-Gibbs, as for the eps move.
function shear_move!(model::covid4, U_NK, V_KM, b_mu, nattempt::Int)
    nacc = 0; ntry = 0
    thk = Vector{Float64}(undef, model.N)
    phj = Vector{Float64}(undef, model.M)
    @inbounds for j in 1:model.K, k in 1:model.K
        j == k && continue
        lp_phj_cur = logprior_phi_vec(model, view(V_KM, j, :))
        for _ in 1:nattempt
            ntry += 1
            # Feasible interval for a. a > 0 shrinks phi_j (needs a < phi_jt/phi_kt for all
            # t); a < 0 shrinks theta_k (needs |a| < theta_ik/theta_ij for all i). The minima
            # over 295 times and N counties make this interval NARROW, so a fixed-width
            # Gaussian lands outside it almost always (measured: 0 of 1600 accepted). A
            # uniform draw on the interval is symmetric -- a shear is a translation along
            # the line, so the interval after the move is the old one shifted by a, same
            # width -- and it reaches the whole feasible range in one step.
            ahi = Inf; alo = -Inf
            for t in 1:model.M
                r = V_KM[j,t] / V_KM[k,t]; r < ahi && (ahi = r)
            end
            for i in 1:model.N
                r = -U_NK[i,k] / U_NK[i,j]; r > alo && (alo = r)
            end
            (isfinite(ahi) && isfinite(alo) && ahi > alo) || continue
            a = alo + (ahi - alo) * rand()
            a *= 0.999   # stay strictly inside the cone
            for i in 1:model.N; thk[i] = U_NK[i,k] + a*U_NK[i,j]; end
            for t in 1:model.M; phj[t] = V_KM[j,t] - a*V_KM[k,t]; end
            lp_phj_new = logprior_phi_vec(model, phj)
            isfinite(lp_phj_new) || continue
            lpt = 0.0
            for i in 1:model.N
                lpt += (model.a - 1)*(log(thk[i]) - log(U_NK[i,k])) - b_mu*(thk[i] - U_NK[i,k])
            end
            if log(rand()) < lpt + (lp_phj_new - lp_phj_cur)
                for i in 1:model.N; U_NK[i,k] = thk[i]; end
                for t in 1:model.M; V_KM[j,t] = phj[t]; end
                lp_phj_cur = lp_phj_new
                nacc += 1
            end
        end
    end
    return nacc / max(ntry, 1)
end

# Univariate slice sampler (Neal 2003, stepping out then shrinkage) on a log density.
function slice1d(logf, x0::Float64, w::Float64; maxsteps::Int=20)
    f0 = logf(x0)
    # A current point with zero density (e.g. a trajectory entry that underflowed to 0.0
    # in a Gamma draw with a tiny shape) would make the shrinkage loop spin forever, since
    # no proposal can exceed -Inf. Leave the coordinate where it is.
    isfinite(f0) || return x0
    logy = f0 + log(rand())
    L = x0 - w * rand(); R = L + w
    j = floor(Int, maxsteps * rand()); kk = maxsteps - 1 - j
    while j > 0 && logf(L) > logy; L -= w; j -= 1; end
    while kk > 0 && logf(R) > logy; R += w; kk -= 1; end
    while true
        x1 = L + rand() * (R - L)
        logf(x1) > logy && return x1
        x1 < x0 ? (L = x1) : (R = x1)
    end
end

# theta, phi, eps | S, D, Y with the thinning integrated out. The target for each scalar
# is  prior  +  sum_cells [ S log mu - D mu ],  mu_{i,t} = Y_{i,t-1} + eps + sum_k theta phi,
# sampled on the log scale (Jacobian folded into the prior term). G_NM = theta * phi is
# kept current incrementally, so every evaluation is O(M) for a theta, O(N) for a phi.
# Rows of theta are conditionally independent given phi and run in parallel; phi is
# sequential because the gamma-Markov prior couples neighbouring columns.
const SLICE_W = 0.2
# theta_{n,k} | S, D, phi, eps with the thinning integrated out; updates U_NK and G_NM in place.
function theta_slice!(model::covid4, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps_th::Float64, b_mu::Float64)
    N, M, K = model.N, model.M, model.K
    off(n, m) = !model.use_offset ? 0.0 : Float64(m == 1 ? Y0_N[n] : Y_NM[n,m-1])

    # --- theta_{n,k}
    @views Threads.@threads :static for n in 1:N
        @inbounds for k in 1:K
            t0 = U_NK[n,k]
            function lf_theta(x)
                t = exp(x)
                lp = model.a * x - b_mu * t
                dlt = t - t0
                ll = 0.0
                for m in 1:M
                    mu = off(n, m) + eps_th + G_NM[n,m] + dlt * V_KM[k,m]
                    mu > 0 || return -Inf
                    ll += S_NM[n,m] * log(mu) - D_NM[n,m] * mu
                end
                lp + ll
            end
            # A non-positive current value cannot arise in stationarity (Gamma draws and
            # exp(x) are positive) but can from an initialiser -- NMF's multiplicative
            # updates produce exact zeros -- and log(0) = -Inf makes the shrinkage loop
            # spin forever on NaNs. Start such an entry from a small positive value.
            x1 = slice1d(lf_theta, t0 > 0 ? log(t0) : log(1e-8), SLICE_W)
            t1 = exp(x1)
            U_NK[n,k] = t1
            dlt = t1 - t0
            for m in 1:M
                G_NM[n,m] += dlt * V_KM[k,m]
            end
        end
    end
    return nothing
end

# Slice sampling on a bounded interval (L, R) that contains x0: shrinkage only, no
# stepping out. Valid because the initial bracket is the whole support.
function slice_bounded(logf, x0::Float64, L::Float64, R::Float64)
    logy = logf(x0) + log(rand())
    while true
        x1 = L + rand() * (R - L)
        logf(x1) > logy && return x1
        x1 < x0 ? (L = x1) : (R = x1)
    end
end

# theta on the simplex: for each factor k, pair the counties at random and move mass
# delta from one to the other, delta ~ its exact conditional given S, D, phi, eps and the
# Dirichlet(a) prior. Only rows i, i2 of mu change, so a move costs O(M) like the Gamma
# slice step; the pairs of a matching are disjoint, so they run in parallel. Two random
# matchings per sweep give every theta_{i,k} two updates.
function theta_simplex!(model::covid4, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps_th::Float64)
    N, M, K = model.N, model.M, model.K
    off(n, m) = !model.use_offset ? 0.0 : Float64(m == 1 ? Y0_N[n] : Y_NM[n,m-1])
    a = model.a
    npairs = N ÷ 2
    for k in 1:K, _ in 1:2
        perm = randperm(N)
        @views Threads.@threads :static for j in 1:npairs
            i = perm[2j-1]; i2 = perm[2j]
            u = U_NK[i,k]; v = U_NK[i2,k]
            function lf_pair(dlt)
                x = u - dlt; y = v + dlt
                (x > 0 && y > 0) || return -Inf
                lp = (a - 1) * (log(x) + log(y))
                ll = 0.0
                @inbounds for m in 1:M
                    phi = V_KM[k,m]
                    mu1 = off(i, m) + eps_th + G_NM[i,m] - dlt * phi
                    mu2 = off(i2, m) + eps_th + G_NM[i2,m] + dlt * phi
                    (mu1 > 0 && mu2 > 0) || return -Inf
                    ll += S_NM[i,m] * log(mu1) - D_NM[i,m] * mu1 + S_NM[i2,m] * log(mu2) - D_NM[i2,m] * mu2
                end
                lp + ll
            end
            dlt = slice_bounded(lf_pair, 0.0, -v, u)
            U_NK[i,k] = u - dlt; U_NK[i2,k] = v + dlt
            @inbounds for m in 1:M
                G_NM[i,m] -= dlt * V_KM[k,m]; G_NM[i2,m] += dlt * V_KM[k,m]
            end
        end
    end
    return nothing
end

# Factor-scale move for the Dirichlet parametrisation: phi_k <- kappa phi_k with theta fixed.
# On the simplex theta cannot carry a factor's level, so without this the level of phi_k
# only moves through the PGDS step, whose conditional given the thinned counts is far
# narrower than the marginal (the thinning bottleneck relocated). This is a Liu-Sabatti
# scale-group move: kappa is drawn from  pi(kappa phi_k) kappa^(M-1) dkappa, i.e. on
# x = log kappa the target is  log prior(kappa phi_k) + loglik(kappa) + M x,  sampled by
# slice sampling. The likelihood is the exact S-target over all cells (O(NM) per
# evaluation, threaded over columns), so this leaves the posterior invariant.
function phi_scale_move!(model::covid4, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps::Float64)
    N, M, K = model.N, model.M, model.K
    off(n, m) = !model.use_offset ? 0.0 : Float64(m == 1 ? Y0_N[n] : Y_NM[n,m-1])
    G_NM = U_NK * V_KM
    nt = Threads.nthreads()
    part = zeros(Float64, nt)
    row = Vector{Float64}(undef, M)
    for k in 1:K
        row .= V_KM[k, :]
        function lf_scale(x)
            kap = exp(x)
            lp = logprior_phi_vec(model, kap .* row)
            isfinite(lp) || return -Inf
            fill!(part, 0.0)
            @views Threads.@threads :static for m in 1:M
                acc = 0.0
                @inbounds for n in 1:N
                    g = U_NK[n,k] * row[m]
                    mu = off(n, m) + eps + G_NM[n,m] + (kap - 1) * g
                    acc += S_NM[n,m] * log(mu) - D_NM[n,m] * mu
                end
                part[Threads.threadid()] += acc
            end
            lp + sum(part) + M * x
        end
        x1 = slice1d(lf_scale, 0.0, 0.1)
        kap = exp(x1)
        @inbounds for m in 1:M
            G_NM[:, m] .+= (kap - 1) .* U_NK[:, k] .* row[m]
            V_KM[k, m] = kap * row[m]
        end
    end
    return nothing
end

# Multiplicative group moves on a factor's trajectory, against the exact S-target.
#
# The PGDS step redraws phi_k conditional on the thinned counts, which pin every phi_kt to
# ~1/sqrt(count) -- 0.1-0.3% on the full matrix -- so the SHAPE of the trajectory (where the
# wave sits, how wide it is) random-walks with an autocorrelation of thousands of sweeps
# even though mu and theta mix. The scale move handles the one global direction phi_k -> kappa
# phi_k; this handles the next two: phi_kt -> phi_kt exp(a g(t)) with g = tilt (t - tbar)/M,
# which moves the centre of mass, and g = curvature ((t - tbar)/M)^2 - mean, which changes
# the width. Each is an elementwise bijection with log-Jacobian a * sum_t g(t), and a is
# drawn by slice sampling from its exact conditional: gamma-Markov prior of the deformed
# row + likelihood sum_it [S log mu - D mu] over all cells (O(NM) per evaluation, threaded
# over columns) + the Jacobian. theta and everything else are fixed. Placed after the theta
# step, where the thinned counts have been discarded, so it is ordinary slice-within-Gibbs
# on the posterior; the PGDS pass, thinning, CRT and order-statistic steps are untouched.
function phi_group_move!(model::covid4, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps::Float64)
    N, M, K = model.N, model.M, model.K
    off(n, m) = !model.use_offset ? 0.0 : Float64(m == 1 ? Y0_N[n] : Y_NM[n,m-1])
    G_NM = U_NK * V_KM
    # smooth multiplicative basis: cosines of order 1..J on [0, 1] (order 1 ~ tilt, 2 ~
    # curvature, higher orders reshape the wave more locally), each mean-zero
    basis = [[cos(j * pi * (t - 0.5) / M) for t in 1:M] for j in 1:PHI_GROUP_J[]]
    nt = Threads.nthreads(); part = zeros(Float64, nt)
    row = Vector{Float64}(undef, M); newrow = Vector{Float64}(undef, M)
    for k in 1:K, g in basis
        row .= V_KM[k, :]
        # entries that underflowed to 0.0 (Gamma draws with shape ~d) have no finite log
        # prior; the gamma-Markov chain keeps such tails tiny anyway, so floor them.
        @inbounds for t in 1:M; row[t] = max(row[t], 1e-200); end
        sumg = sum(g)
        function lf(a)
            @inbounds for t in 1:M; newrow[t] = row[t] * exp(a * g[t]); end
            lp = logprior_phi_vec(model, newrow)
            isfinite(lp) || return -Inf
            fill!(part, 0.0)
            @views Threads.@threads :static for m in 1:M
                d = newrow[m] - row[m]
                acc = 0.0
                @inbounds for n in 1:N
                    mu = off(n, m) + eps + G_NM[n,m] + U_NK[n,k] * d
                    acc += S_NM[n,m] * log(mu) - D_NM[n,m] * mu
                end
                part[Threads.threadid()] += acc
            end
            lp + sum(part) + a * sumg
        end
        a1 = slice1d(lf, 0.0, 0.05)
        @inbounds for t in 1:M
            nv = row[t] * exp(a1 * g[t]); d = nv - row[t]
            V_KM[k, t] = nv
            for n in 1:N; G_NM[n,t] += U_NK[n,k] * d; end
        end
    end
    return nothing
end

function collapsed_update!(model::covid4, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps::Float64, b_mu::Float64)
    N, M, K = model.N, model.M, model.K
    G_NM = U_NK * V_KM
    off(n, m) = !model.use_offset ? 0.0 : Float64(m == 1 ? Y0_N[n] : Y_NM[n,m-1])
    # `eps` is reassigned below; closures capturing a reassigned local get a Core.Box and
    # dynamic dispatch in the innermost loop, so each stage binds it to a fresh name.
    eps_th = eps
    model.dir_theta ? theta_simplex!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps_th) :
                      theta_slice!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps_th, b_mu)
    if model.dir_theta
        phi_scale_move!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps_th)
        G_NM .= U_NK * V_KM
    end
    if PHI_GROUP[]
        phi_group_move!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps_th)
        G_NM .= U_NK * V_KM
    end

    # --- eps (one scalar; the N*M sum is threaded)
    nt = Threads.nthreads()
    part = zeros(Float64, nt)
    function lf_eps(x)
        e = exp(x)
        fill!(part, 0.0)
        @views Threads.@threads :static for m in 1:M
            acc = 0.0
            @inbounds for n in 1:N
                mu = off(n, m) + e + G_NM[n,m]
                acc += S_NM[n,m] * log(mu) - D_NM[n,m] * mu
            end
            part[Threads.threadid()] += acc
        end
        model.g * x - model.h * e + sum(part)
    end
    eps_new = exp(slice1d(lf_eps, log(eps_th), SLICE_W))

    # --- phi_{k,m}: column m's conditional depends on its own column of mu and on the
    # neighbouring columns m-1, m+1 through the gamma-Markov prior, so alternate columns
    # are conditionally independent: all odd m in parallel, then all even m.
    c, d = model.c, model.d
    for parity in (1, 0)
        @views Threads.@threads :static for m in (1 + parity):2:M
            @inbounds for k in 1:K
                t0 = V_KM[k,m]
                iid = model.iid_phi
                shape_own = (m == 1 || iid) ? model.start_V1 : c * V_KM[k,m-1] + d
                rate_own  = (m == 1 || iid) ? model.start_V2 : c
                vnext = m < M ? V_KM[k,m+1] : 0.0
                function lf_phi(x)
                    t = exp(x)
                    lp = shape_own * x - rate_own * t
                    if m < M && !iid
                        sh = c * t + d
                        lp += sh * log(c) - SpecialFunctions.loggamma(sh) + (sh - 1) * log(vnext)
                    end
                    dlt = t - t0
                    ll = 0.0
                    for n in 1:N
                        mu = off(n, m) + eps_new + G_NM[n,m] + dlt * U_NK[n,k]
                        mu > 0 || return -Inf
                        ll += S_NM[n,m] * log(mu) - D_NM[n,m] * mu
                    end
                    lp + ll
                end
                x1 = slice1d(lf_phi, t0 > 0 ? log(t0) : log(1e-8), SLICE_W)
                t1 = exp(x1)
                V_KM[k,m] = t1
                dlt = t1 - t0
                for n in 1:N
                    G_NM[n,m] += dlt * U_NK[n,k]
                end
            end
        end
    end
    return eps_new
end

function backward_sample(model::covid4, data, state, mask=nothing; skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Beta_NQ = copy(state["Beta_NQ"])
    Tau_QM = copy(state["Tau_QM"])
    D_NM = Int.(copy(state["D_NM"]))
    sigma2_beta = copy(state["sigma2_beta"])
    sigma2_tau = copy(state["sigma2_tau"])

    eps = copy(state["eps"])
    b_mu = model.hier_b ? copy(state["b_mu"])::Float64 : model.b

    Y0_N = copy(state["Y0_N"])

    # skipupdate = ["U_NK"] freezes the whole theta / phi / eps section (latent sums,
    # thinning or collapsed update, scale move) so the D / PG / tau / beta layer can be
    # tested on its own; still a valid partial Gibbs kernel.
    skip_tp = !isnothing(skipupdate) && ("U_NK" in skipupdate)
    scale_acc = 0.0
    # Loop over the non-zeros in Y_NM and allocate
    nt = Threads.nthreads()
    sum_noise_thr  = zeros(Float64, nt)
    Y_NK_thr = [zeros(Int, model.N, model.K) for _ in 1:nt]
    Y_MK_thr = [zeros(Int, model.M, model.K) for _ in 1:nt]
    P_K_thr = [zeros(Float64, model.K + 2) for _ in 1:nt]
    ykbuf_thr = [zeros(Int, model.K + 2) for _ in 1:nt]
    S_NM = model.collapsed ? Matrix{Int}(undef, model.N, model.M) : Matrix{Int}(undef, 0, 0)
    # :static is required, not cosmetic -- the default :dynamic schedule may migrate a
    # task between threads mid-iteration, which would make threadid() name a different
    # accumulator partway through and silently corrupt the sufficient statistics.
    if !skip_tp
    @views @threads :static for n in 1:model.N
        @views for m in 1:model.M
            tid = Threads.threadid()
            Ylast = !model.use_offset ? 0 : (m == 1 ? Y0_N[n] : Y_NM[n,m-1])
            P_K = P_K_thr[tid]
            @inbounds begin
                @simd for k in 1:model.K
                    P_K[k] =  U_NK[n,k] * V_KM[k,m]
                end
                P_K[model.K+1] = Ylast
                P_K[model.K+2] = eps
            end
            mu = sum(P_K)
            D = D_NM[n,m]; j = div(D,2) + 1
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(OrderStatistic(Poisson(mu), D, j))
                end
            end
            Z = sampleSumGivenOrderStatistic(Y_NM[n, m], D, j, Poisson(mu))
            model.collapsed && (S_NM[n,m] = Z)
            if (!model.collapsed || model.phi_da) && Z > 0
                y_k = thin_multinomial!(ykbuf_thr[tid], Z, P_K, model.K + 2)
                @inbounds begin
                    sum_noise_thr[tid]  += y_k[model.K+2]
                end
                @inbounds for k in 1:model.K
                    Y_NK_thr[tid][n, k] += y_k[k]
                    Y_MK_thr[tid][m, k] += y_k[k]
                end
            end
        end
    end

    sum_noise  = sum(sum_noise_thr)
    Y_NK  = sum(Y_NK_thr)
    Y_MK  = sum(Y_MK_thr)

    # D_NM as Float64 once, so the two products it enters below go through BLAS instead
    # of a generic mixed Int/Float fallback.
    Df_NM = Float64.(D_NM)

    if model.collapsed && !model.phi_da
        eps = collapsed_update!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps, b_mu)
        if model.hier_b
            b_mu = sample_rate_hyper(U_NK, model.a)
        end
    else
    theta_last = model.collapsed && model.phi_da   # theta by the collapsed slice step, after phi
    #update county factors
    # post_rate[n,k] = b + sum_m D[n,m] V[k,m] is exactly (D_NM * V_KM')[n,k]; as a single
    # gemm rather than N*K length-M broadcasts, each of which allocated a temporary.
    if !theta_last
    DV_NK = Df_NM * transpose(V_KM)
    @views for k in 1:model.K
        @views for n in 1:model.N
            post_rate = b_mu + DV_NK[n,k]
            post_shape = model.a + Y_NK[n,k]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    # population rate for theta | - : conjugate, and the whole of the hierarchical variant
    if model.hier_b
        b_mu = sample_rate_hyper(U_NK, model.a)
    end
    end

    #update noise term
    post_shape = model.g + sum_noise
    post_rate = model.h + sum(D_NM)
    eps = rand(Gamma(post_shape, 1/post_rate))

    #update time factors
    C1_KM = transpose(U_NK) * Df_NM

    # l_KM holds CRT/binomial counts, so it is integer-valued; it used to be a Float64
    # matrix, which forced the CRT count through a float and back.
    l_KM = zeros(Int, model.K, model.M+1)
    q_KM = zeros(model.K,model.M+1)

    #backward pass (skipped under the iid prior: no chain to pass information along)
    @views for m in (model.iid_phi ? (2:1) : (model.M:-1:2))
        @views for k in 1:model.K
            q_KM[k,m] = log(1 + (C1_KM[k,m]/model.c) + q_KM[k,m+1])

            temp = sampleCRTlecam(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)

            l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
        end
    end
    
    @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    #forward pass
    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1 || model.iid_phi
                V_KM[k,m] = rand(Gamma(model.start_V1 + Y_MK[m,k] + l_KM[k,m+1], 1/(model.start_V2 + C1_KM[k,m] + model.c*q_KM[k,m+1])))
            else
                V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C1_KM[k,m] + model.c*q_KM[k,m+1])))
            end 
        end 
    end 
    if theta_last
        G_NM = U_NK * V_KM
        model.dir_theta ? theta_simplex!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps) :
                          theta_slice!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, G_NM, eps, b_mu)
        model.dir_theta && phi_scale_move!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps)
        PHI_GROUP[] && phi_group_move!(model, Y_NM, Y0_N, S_NM, D_NM, U_NK, V_KM, eps)
        if model.hier_b
            b_mu = sample_rate_hyper(U_NK, model.a)
        end
    end
    end # collapsed / thinning branch
    
    # Interleaved scale move on (theta_{:,k}, phi_{k,:}). Valid here because it targets
    # the same conditional the theta and phi updates just sampled from, and it leaves
    # mu_NM -- and so the D update below -- untouched by construction.
    scale_acc = (model.scale_move && !model.dir_theta) ? scale_move!(model, U_NK, V_KM, b_mu, 10) : 0.0
    end # skip_tp

    #Polya-gamma augmentation to update D, Beta and Tau
    if isnothing(skipupdate) || !("D_NM" in skipupdate)
        # inside the branch: with D pinned none of this is needed, and mu_NM alone is an
        # N x M x K product over 900k cells.
        mu_NM = U_NK * V_KM
        F_NM = Beta_NQ * Tau_QM
        W_NM = zeros(Float64, model.N, model.M)
        p_NM = logistic.(F_NM)
        # Dmax is odd, so the Polya-Gamma shape b = (Dmax-1)/2 is a positive integer and
        # the hybrid sampler always dispatches to Devroye for the sizes used here. Calling
        # it directly hoists a struct construction and four string comparisons out of
        # ~900k draws per sweep; it is the same routine on the same stream.
        bpg = (model.Dmax - 1) ÷ 2
        pg_devroye = bpg >= 1 && bpg < 13
        # Threaded over m with n inner, which is the order the previous flat idx loop ran
        # in, so a single-threaded run consumes the RNG stream identically.
        @views @threads :static for m in 1:model.M
            w = DUpdateWork(model)
            @views for n in 1:model.N
                Ylast = !model.use_offset ? 0 : (m == 1 ? Y0_N[n] : Y_NM[n,m-1])
                mu = Ylast + eps + mu_NM[n,m]

                D_NM[n,m] = update_D(model, Y_NM[n,m], mu, p_NM[n,m], w)
                W_NM[n,m] = pg_devroye ? rand_pgdevroye(bpg, F_NM[n,m], Random.default_rng()) :
                                         rand(PolyaGammaHybridSampler((model.Dmax - 1)/2, F_NM[n,m]))
            end
        end


        # kappa_{n,m} = (D_{n,m}-1)/2 - (Dmax-1)/4, the Polya-Gamma pseudo-response, plus
        # transposes of it and of W. The Tau block walks columns of the N x M layout and
        # the Beta block walks rows; materialising both orientations once, in parallel,
        # makes every inner loop below contiguous.
        c0 = (model.Dmax - 1)/4
        Kap_NM = Matrix{Float64}(undef, model.N, model.M)
        Wt_MN  = Matrix{Float64}(undef, model.M, model.N)
        Kapt_MN = Matrix{Float64}(undef, model.M, model.N)
        @views @threads :static for m in 1:model.M
            @inbounds for n in 1:model.N
                kap = (D_NM[n,m] - 1)/2 - c0
                Kap_NM[n,m] = kap
                Kapt_MN[m,n] = kap
                Wt_MN[m,n] = W_NM[n,m]
            end
        end

        # --- tau_m | - : the DRAW is sequential in m, deliberately.
        #
        # tau_{q,m} has the AR(1) prior of Eq. (14), so its full conditional depends on
        # BOTH neighbours, tau_{.,m-1} and tau_{.,m+1}. This loop used to run under
        # @threads, which was wrong twice over: it is a data race (threads read the
        # neighbour columns other threads are writing), and even without the race,
        # updating all columns from the same old state is a Jacobi sweep, not a Gibbs
        # sweep, so it does not leave the posterior invariant.
        #
        # Only the draw has to be serial, though. The posterior precision
        # A_m = Beta' diag(w_m) Beta + lambda I and the data part of the mean
        # Beta' kappa_m do not involve Tau at all, and they are all of the arithmetic
        # (N*Q*M multiply-adds against a Q x Q Cholesky). So they are built and factored
        # in parallel over m first, and the sequential scan that follows only adds the
        # two neighbour terms and solves a 25 x 25 triangular system per column.
        lam_own  = 1/sigma2_tau
        lam_next = model.tauc^2 / sigma2_tau
        chol_M   = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, model.M)
        rhsbase_QM = Matrix{Float64}(undef, model.Q, model.M)
        BW_NQ_thr = [Matrix{Float64}(undef, model.N, model.Q) for _ in 1:nt]
        @views @threads :static for m in 1:model.M
            BW_NQ = BW_NQ_thr[Threads.threadid()]
            @inbounds for q in 1:model.Q
                @simd for n in 1:model.N
                    BW_NQ[n,q] = Beta_NQ[n,q] * W_NM[n,m]
                end
            end
            A_QQ = Matrix{Float64}(undef, model.Q, model.Q)
            mul!(A_QQ, transpose(BW_NQ), Beta_NQ)
            lam = m == model.M ? lam_own : lam_own + lam_next
            @inbounds for q in 1:model.Q
                A_QQ[q,q] += lam
            end
            chol_M[m] = cholesky!(Symmetric(A_QQ))
            mul!(view(rhsbase_QM, :, m), transpose(Beta_NQ), view(Kap_NM, :, m))
        end

        rhs_Q = Vector{Float64}(undef, model.Q)
        z_Q   = Vector{Float64}(undef, model.Q)
        if TAU_NAIVE[]
            @views for m in 1:model.M
                A = transpose(Beta_NQ) * (W_NM[:, m] .* Beta_NQ) + (m == model.M ? lam_own : lam_own + lam_next) * I
                r = transpose(Beta_NQ) * Kap_NM[:, m]
                for q in 1:model.Q
                    tprev = m == 1 ? model.start_tau : Tau_QM[q,m-1]
                    r[q] += lam_own * (model.taud + model.tauc*tprev)
                    m < model.M && (r[q] += (model.tauc/sigma2_tau) * (Tau_QM[q,m+1] - model.taud))
                end
                Sig = Symmetric(inv(Matrix(A)))
                Tau_QM[:, m] = rand(MvNormal(Sig * r, Sig))
            end
        end
        @views for m in (TAU_NAIVE[] ? (1:0) : (1:model.M))
            @inbounds for q in 1:model.Q
                tprev = m == 1 ? model.start_tau : Tau_QM[q,m-1]
                r = rhsbase_QM[q,m] + lam_own * (model.taud + model.tauc*tprev)
                if m < model.M
                    r += (model.tauc/sigma2_tau) * (Tau_QM[q,m+1] - model.taud)
                end
                rhs_Q[q] = r
            end
            # Draw from N(A^{-1} rhs, A^{-1}) through the Cholesky factor of the
            # PRECISION: x = A^{-1}rhs + U^{-1} z with A = U'U. Same distribution as
            # rand(MvNormal(inv(A)*rhs, inv(A))) without forming the inverse.
            Fc = chol_M[m]
            ldiv!(Fc, rhs_Q)              # rhs_Q <- A^{-1} rhs, the posterior mean
            randn!(z_Q)
            ldiv!(Fc.U, z_Q)
            @inbounds for q in 1:model.Q
                Tau_QM[q,m] = rhs_Q[q] + z_Q[q]
            end
        end

        # --- beta_n | - : rows ARE conditionally independent given Tau, W and D, so this
        # one is genuinely parallel.
        # Per-thread scratch, allocated once rather than once per county: these four
        # buffers inside the loop body were ~200 MB of the sweep's allocations.
        TW_QM_thr  = [Matrix{Float64}(undef, model.Q, model.M) for _ in 1:nt]
        An_QQ_thr  = [Matrix{Float64}(undef, model.Q, model.Q) for _ in 1:nt]
        rhs_n_thr  = [Vector{Float64}(undef, model.Q) for _ in 1:nt]
        z_Q_thr    = [Vector{Float64}(undef, model.Q) for _ in 1:nt]
        @views @threads :static for n in 1:model.N
            tid = Threads.threadid()
            TW_QM  = TW_QM_thr[tid]
            An_QQ  = An_QQ_thr[tid]
            rhs_n  = rhs_n_thr[tid]
            # Wt_MN[:,n] is county n's weights contiguously; W_NM[n,:] would be a
            # stride-N walk of a 7 MB matrix, once per county.
            @inbounds for m in 1:model.M
                wnm = Wt_MN[m,n]
                @simd for q in 1:model.Q
                    TW_QM[q,m] = Tau_QM[q,m] * wnm
                end
            end
            kbuf_M = view(Kapt_MN, :, n)
            mul!(An_QQ, TW_QM, transpose(Tau_QM))
            @inbounds for q in 1:model.Q
                An_QQ[q,q] += 1/sigma2_beta
            end
            mul!(rhs_n, Tau_QM, kbuf_M)
            Fc = cholesky!(Symmetric(An_QQ))
            ldiv!(Fc, rhs_n)              # rhs_n <- A^{-1} rhs, the posterior mean
            z = z_Q_thr[tid]
            randn!(z)
            ldiv!(Fc.U, z)
            @inbounds for q in 1:model.Q
                Beta_NQ[n,q] = rhs_n[q] + z[q]
            end
        end

        mu1 = model.tauc*model.start_tau + model.taud
        sse = sum((Tau_QM[:,1] .- mu1).^2)

        for m in 2:model.M
            sse += sum((Tau_QM[:,m] .- model.taud .-
                        model.tauc*Tau_QM[:,m-1]).^2)
        end

        sigma2_tau = rand(InverseGamma(
            model.alpha0 + model.Q*model.M/2,
            model.beta0 + sse/2
        ))


        sigma2_beta = rand(InverseGamma(
            model.alpha0 + (model.Q*model.N)/2,
            model.beta0 + sum(Beta_NQ.^2)/2
        ))
    end

    # Marginal Metropolis step on eps, at the END of the sweep: the latent counts have been
    # used and discarded, so the state here is just the parameters and this is ordinary
    # Metropolis-within-Gibbs on pi(eps | theta, phi, D, Y).
    shear_acc = model.shear_move ? shear_move!(model, U_NK, V_KM, b_mu, 4) : 0.0

    eps_acc = 0.0
    if model.eps_mh
        works = [DUpdateWork(model) for _ in 1:nt]
        muNM = U_NK * V_KM
        eps, eps_acc = eps_marginal_move!(model, Y_NM, Y0_N, muNM, D_NM, eps, works, 4, 0.15)
    end

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "eps" => eps, "b_mu" => b_mu,
    "scale_acc" => scale_acc, "eps_acc" => eps_acc, "shear_acc" => shear_acc,
    "D_NM"=>D_NM, "Tau_QM"=>Tau_QM, "Beta_NQ"=>Beta_NQ,
     "sigma2_beta"=>sigma2_beta, "sigma2_tau"=>sigma2_tau,
    "Y0_N"=>Y0_N,)
    #"Y_NMKplus2"=>Y_NMKplus2)
    return data, state
end

#forecasting code 

function predict(model::covid4, state, info, n, m, Ylast)
    eps = state["eps"]
    U_NK = state["U_NK"]
    V_KM = state["V_KM"]
    D_NM = state["D_NM"]
    return rand(OrderStatistic(Poisson(Ylast + eps + sum(U_NK[n,:] .* V_KM[:,m])), D_NM[n,m],div(D_NM[n,m],2)+1))
end

function predict_x(model::covid4, state, info, n, mstart, Ystart, x)
    result = zeros(x)
    Ylast = Ystart
    m = mstart + 1
    for i in 1:length(result)
        result[i] = predict(model, state, info, n, m, Ylast)
        m += 1
        Ylast = result[i]
    end
    return result
end


function forecast(model::covid4, state, data, info, Ti)
    lastgamma = state["V_KM"][:,end]
    forecastGamma_KTi = zeros(model.K, Ti)
    for i in 1:Ti
        forecastGamma_KTi[:,i] = rand.(Gamma.(model.c*lastgamma .+ model.d, 1/model.c))
        lastgamma = copy(forecastGamma_KTi[:,i])
    end 
    Ylast = data["Y_NM"][:,end]

    mu_NTi = state["U_NK"] * forecastGamma_KTi

    Y_NTi = zeros(model.N,Ti)
    for i in 1:Ti
        mu = Ylast .+ state["eps"] .+ mu_NTi[:,i]
        Y_NTi[:,i] = rand.(OrderStatistic.(Poisson.(mu), model.D, model.j))
        Ylast = copy(Y_NTi[:,i])
    end
    return Y_NTi
end