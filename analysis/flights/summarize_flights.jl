# Section 7.1 results table: information gain, MCMC diagnostics, and computing performance.
#
# Reads the per-run .jld files written by runflights_heldout.jl and produces
#   Table 2      -- information gain over the D=1 Poisson baseline, mean +/- 1.96 SE
#                   across the five train/test splits (referee R1 1(a) asked for the
#                   convention to be stated; it is the SE over splits)
#   Table 2b     -- computing performance (referee R1 2(b)): wall-clock, ESS, ESS/s,
#                   and seconds per 1000 effective samples, on a NAMED functional
#
# ESS/Rhat use rank-normalized split-Rhat and bulk/tail ESS (Vehtari et al. 2021) via
# MCMCDiagnosticTools, computed on functionals that mean the same thing in every model:
#   * the held-out log predictive density (the quantity information gain is built from)
#   * mu_k is reported separately and labelled, because it is NOT comparable across
#     models -- STAR's mu_k is a latent Gaussian mean on the transformed scale.
#
# usage: julia --project=. analysis/flights/summarize_flights.jl
using JLD, Statistics, Printf, LogExpFunctions
using MCMCDiagnosticTools

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DIR  = joinpath(ROOT, "output/flights/revisionsamples")

const MASKS  = [101, 102, 103, 104, 105]
const CHAINS = [1, 2, 3, 4]
# (D, type, g, label, baseline-key). Each family is scored against its OWN D=1 baseline so
# the hierarchical and flat comparisons are each internally consistent.
const CONFIGS = [
    (0, 1, 0, "MedPois hier, D inferred", :hier),
    (3, 1, 0, "MedPois hier, D = 3",      :hier),
    (5, 1, 0, "MedPois hier, D = 5",      :hier),
    (7, 1, 0, "MedPois hier, D = 7",      :hier),
    (9, 1, 0, "MedPois hier, D = 9",      :hier),
    (0, 2, 2, "STAR hier (sqrt)",         :hier),
    (0, 5, 0, "MedNB, D inferred",        :hier),
]
const BASELINES = Dict(:hier => (1, 1, 0))

fname(D, t, g, m, c) = t == 1 ? "MedPoissonD$(D)mask$(m)chain$(c).jld" :
                       t == 2 ? "STARg$(g)mask$(m)chain$(c).jld" :
                       t == 3 ? "MedPoissonFlatD$(D)mask$(m)chain$(c).jld" :
                       t == 5 ? "MedNBD$(D)mask$(m)chain$(c).jld" :
                                "STARFlatg$(g)mask$(m)chain$(c).jld"
loadrun(D, t, g, m, c) = load(joinpath(DIR, fname(D, t, g, m, c)))

# Pooled held-out information rate for one split, combining all four chains into a single
# 2000-draw posterior. Each run stored log( mean_s p(y_h | theta_s) ) over its own 500
# draws, so pooling is logsumexp over chains minus log(nchains) -- exact, not an average
# of per-chain rates.
function inforate_pooled(D, t, g, mask)
    percell = [loadrun(D, t, g, mask, c)["heldout_logmeanexp_cells"] for c in CHAINS]
    H = length(percell[1])
    pooled = [logsumexp([percell[ci][h] for ci in eachindex(CHAINS)]) - log(length(CHAINS))
              for h in 1:H]
    return mean(pooled)
end

# (draws, chains) matrix of the per-draw held-out log predictive density
function loglik_matrix(D, t, g, mask)
    cols = [loadrun(D, t, g, mask, c)["heldout_loglik_draws"] for c in CHAINS]
    return reduce(hcat, cols)
end

# (draws, chains, routes) array of mu_k
function mu_array(D, t, g, mask)
    key = t == 5 ? "r_R" : "U_R"     # MedNB stores the NB size as r_R, not U_R
    per = [reduce(hcat, [s[key] for s in loadrun(D, t, g, mask, c)["samples"]])' for c in CHAINS]
    S, R = size(per[1])
    A = Array{Float64}(undef, S, length(CHAINS), R)
    for (ci, M) in enumerate(per); A[:, ci, :] = M; end
    return A
end

safe(f, x) = try f(x) catch; NaN end
ess_bulk(x) = safe(y -> ess(y; kind = :bulk), x)
ess_tail(x) = safe(y -> ess(y; kind = :tail), x)

function summarize()
    println("="^100)
    @printf("Section 7.1 -- %d splits x %d chains, %d retained draws per chain\n",
            length(MASKS), length(CHAINS), length(loadrun(0,1,0,MASKS[1],1)["heldout_loglik_draws"]))
    meta = loadrun(0, 1, 0, MASKS[1], 1)
    @printf("Hardware: %s, %d cores, %d thread(s) per run, node %s; Julia %s; commit %s\n",
            meta["cpu"], meta["ncores"], meta["nthreads"], meta["hostname"], meta["julia"], meta["gitcommit"])
    println("All models timed single-threaded on the same node. Timing excludes JIT warm-up")
    println("and held-out evaluation (recorded separately).")

    # ---------------- Table 2: information gain ----------------
    println("\n" * "="^100)
    println("TABLE 2 -- information gain over the D=1 Poisson baseline (higher is better)")
    println("mean +/- 1.96 SE across the five train/test splits\n")
    @printf("%-28s %10s %10s %10s\n", "model", "IG", "1.96*SE", "info rate")
    base = Dict(k => Dict(m => inforate_pooled(v..., m) for m in MASKS) for (k, v) in BASELINES)
    results = Dict{String,Any}()
    for (D, t, g, lab, fam) in CONFIGS
        igs = [inforate_pooled(D, t, g, m) - base[fam][m] for m in MASKS]
        irs = [inforate_pooled(D, t, g, m) for m in MASKS]
        se = std(igs) / sqrt(length(igs))
        results[lab] = (ig = mean(igs), se = se)
        @printf("%-28s %10.4f %10.4f %10.4f\n", lab, mean(igs), 1.96 * se, mean(irs))
    end
    for (fam, key) in BASELINES
        @printf("%-28s %10.4f %10s %10.4f\n",
                "  (baseline $(fam), D = 1)", 0.0, "--", mean(values(base[fam])))
    end

    # ---------------- Table 2b: computing performance ----------------
    println("\n" * "="^100)
    println("TABLE 2b -- computing performance (referee 2(b))")
    println("ESS/Rhat on the HELD-OUT LOG PREDICTIVE DENSITY: identifiable, and the quantity")
    println("information gain is built from, so it is comparable across all models.")
    println("ESS is the total from 4 chains; cost is the 4-chain wall-clock.\n")
    @printf("%-28s %8s %9s %7s %8s %8s %7s %9s %11s\n",
            "model", "s/chain", "s/100it", "-Inf%", "ESSbulk", "ESStail", "Rhat", "ESS/s", "s/1000ESS")
    allcfg = vcat(CONFIGS, [(1, 1, 0, "MedPois hier, D = 1", :hier)])
    for (D, t, g, lab, _) in allcfg
        times = [loadrun(D, t, g, m, c)["time"] for m in MASKS for c in CHAINS]
        tmed = median(times)
        nsweeps = loadrun(D, t, g, MASKS[1], 1)["nsweeps"]
        eb = Float64[]; et = Float64[]; rh = Float64[]; ninf = 0; ntot = 0
        for m in MASKS
            X = loglik_matrix(D, t, g, m)
            ninf += count(!isfinite, X); ntot += length(X)
            A = reshape(X, size(X, 1), size(X, 2), 1)
            push!(eb, ess_bulk(A)[1]); push!(et, ess_tail(A)[1]); push!(rh, safe(rhat, A)[1])
        end
        essb = mean(eb); cost4 = 4 * tmed     # four chains is what buys that ESS
        pinf = 100 * ninf / ntot
        if ninf > 0
            # The functional is -Inf whenever ANY held-out flight gets zero predictive mass
            # under that draw, so ESS/Rhat computed on it are meaningless. Report the
            # incidence instead of a number that looks like a diagnostic but is not.
            @printf("%-28s %8.1f %9.3f %6.1f%% %8s %8s %7s %9s %11s\n",
                    lab, tmed, 100 * tmed / nsweeps, pinf, "--", "--", "--", "--", "--")
        else
            @printf("%-28s %8.1f %9.3f %6.1f%% %8.0f %8.0f %7.3f %9.2f %11.1f\n",
                    lab, tmed, 100 * tmed / nsweeps, pinf, essb, mean(et), maximum(rh),
                    essb / cost4, 1000 * cost4 / essb)
        end
    end
    println("\n-Inf% = share of (draw, chain) values of the functional that are -Inf, i.e. draws")
    println("under which at least one held-out flight receives zero predictive probability.")
    println("Where that occurs, ESS/Rhat on this functional are not computable and are shown")
    println("as '--'; information gain is unaffected, since no CELL is -Inf across all draws.")

    # ---------------- decomposition + model-specific functional ----------------
    println("\n" * "="^100)
    println("DECOMPOSITION -- ESS/s separates into how well the sampler mixes per draw")
    println("and how fast draws are produced. mu_k ESS is MODEL-SPECIFIC and not")
    println("comparable across families (STAR's mu_k is on the transformed scale).\n")
    @printf("%-28s %14s %14s %16s\n", "model", "ESS/draw", "draws/s", "median ESS(mu_k)")
    for (D, t, g, lab, _) in CONFIGS
        times = [loadrun(D, t, g, m, c)["time"] for m in MASKS for c in CHAINS]
        tmed = median(times)
        ndraws = loadrun(D, t, g, MASKS[1], 1)["nsamples"]
        eb = Float64[]; emu = Float64[]; ninf = 0
        for m in MASKS
            X = loglik_matrix(D, t, g, m)
            ninf += count(!isfinite, X)
            push!(eb, ess_bulk(reshape(X, size(X, 1), size(X, 2), 1))[1])
            v = ess_bulk(mu_array(D, t, g, m))
            push!(emu, median(filter(isfinite, v)))
        end
        # same caveat as above: ESS/draw is not computable when the functional is -Inf
        essdraw = ninf > 0 ? "--" : @sprintf("%.3f", mean(eb) / (4 * ndraws))
        @printf("%-28s %14s %14.2f %16.0f\n", lab, essdraw, ndraws / tmed, mean(emu))
    end
    println("="^100)
end

summarize()
