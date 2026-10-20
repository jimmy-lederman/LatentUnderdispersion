# Section 7.2 results: forecasting information gain, MCMC diagnostics, and computing
# performance for the COVID-19 case study.
#
#   julia --project=. analysis/covid/summarize_covid.jl
#
# Reads the per-run .jld files written by runcovid_heldout.jl and produces
#   Table E.x   -- information gain over the D = 1 Poisson baseline, mean +/- 1.96 SE
#                  across the train/test splits
#   Table E.y   -- MCMC diagnostics (referee R1 2(d)) and computing performance (R1 2(b))
#
# ESS/Rhat use rank-normalized split-Rhat and bulk/tail ESS (Vehtari et al. 2021) via
# MCMCDiagnosticTools, on functionals that are IDENTIFIABLE and mean the same thing in
# every model, exactly as in analysis/flights/summarize_flights.jl:
#   * the held-out log predictive density (the quantity information gain is built from)
#   * mu_{i,t} on a fixed random subset of 200 cells
#   * mean(D_{i,t}), the average inferred order
# Raw factor entries (theta_{i,k}, phi_{k,t}, beta_{i,q}, tau_{q,t}) are NOT used: they
# are identified only up to the scale trade-off between theta and phi and, for the
# Polya-Gamma layer, up to sign/rotation. That is the same point response_plan WS1.2
# makes for Section 6.3.

using JLD, Statistics, Printf, LogExpFunctions
using MCMCDiagnosticTools

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
# The grid is produced by revision2_experiments/covid/cluster/run_diag_chain.jl, which
# evaluates the held-out likelihood INSIDE the sweep loop and stores functionals instead of
# whole state dicts. That is why the chains are unthinned and every arm shares one set of
# MCMC settings. Its key names differ from the older runcovid_heldout.jl format and it
# stores warm-up sweeps too, so loadrun below maps the names and applies the window.
const DIR  = joinpath(ROOT, "output/covid/infogain")

const MASKS  = [101, 102, 103, 104, 105]
const CHAINS = [1, 2, 3, 4]
# Q = 1 (per-day rho_t, scalar county loading), not the 25 of the original submission:
# Q = 25 produced D-level modes and factor death in 1-2 of every 4 chains, while Q = 1
# removed both and reproduced the submission's reported held-out fit in EVERY chain
# (per-cell -4.338/-4.360/-4.340/-4.345 against -4.346/-4.771/-4.578/-4.338). See
# revision2_experiments/covid/FINDINGS.md.
const K, Q   = 5, 1

# (D, type, label). Every row is scored against the D = 1 row of the same model family.
const CONFIGS = [
    (0, 4, "MedPois, D_{i,t} inferred"),
    (3, 4, "MedPois, D = 3"),
    (5, 4, "MedPois, D = 5"),
    (7, 4, "MedPois, D = 7"),
    (9, 4, "MedPois, D = 9"),
]
const BASELINE = (1, 4, "Poisson (D = 1)")

# The DURABLE companion, not the full chain file: the latter carries a 931 MB per-draw theta
# trace and lives on scratch, which purges. The companion is the same file without that trace
# and holds everything any table or figure is built from.
path(D, t, m, c) = joinpath(DIR, "D$(D)_mask$(m)", "summary_chain$(c).jld")
have(D, t, m, c) = isfile(path(D, t, m, c)) && filesize(path(D, t, m, c)) > 0

# diag key -> the name the rest of this script uses
const KEYMAP = Dict("heldout_loglik_draws" => "ll", "mu_draws" => "mu",
                    "Dmean_draws" => "dm", "time" => "seconds")

function loadrun(D, t, m, c)
    r = load(path(D, t, m, c))
    for (want, have_) in KEYMAP
        !haskey(r, want) && haskey(r, have_) && (r[want] = r[have_])
    end
    # Drop warm-up. run_diag_chain.jl stores EVERY sweep, so without this the per-draw
    # traces would carry burn-in into both the ESS/Rhat figures and the timing column.
    if haskey(r, "sw") && haskey(r, "burnin")
        keep = findall(>(r["burnin"]), r["sw"])
        for k in ("heldout_loglik_draws", "mu_draws", "Dmean_draws")
            haskey(r, k) && (r[k] = r[k][keep])
        end
    end
    return r
end

# Pooled held-out information rate for one split: the four chains form a single posterior,
# so this is a logsumexp over chains per cell, not an average of per-chain rates.
function inforate_pooled(D, t, mask)
    percell = [loadrun(D, t, mask, c)["heldout_logmeanexp_cells"] for c in CHAINS]
    H = length(percell[1])
    return mean(logsumexp([percell[ci][h] for ci in eachindex(CHAINS)]) - log(length(CHAINS))
                for h in 1:H)
end

# (draws, chains) matrix of a per-draw scalar
function draws_matrix(D, t, mask, key)
    return reduce(hcat, [loadrun(D, t, mask, c)[key] for c in CHAINS])
end

# (draws, chains, params) array of mu on the fixed subset of cells
function mu_array(D, t, mask)
    per = [loadrun(D, t, mask, c)["mu_draws"] for c in CHAINS]   # chain -> draw -> cells
    S, P = length(per[1]), length(per[1][1])
    A = Array{Float64}(undef, S, length(CHAINS), P)
    for ci in eachindex(CHAINS), s in 1:S, p in 1:P
        A[s, ci, p] = per[ci][s][p]
    end
    return A
end

# A constant chain makes rhat/ess return NaN rather than throw; keep that visible instead
# of silently dropping it.
safe(f, x) = try f(x) catch; [NaN] end
ess_bulk(x) = safe(y -> ess(y; kind = :bulk), x)
ess_tail(x) = safe(y -> ess(y; kind = :tail), x)

function complete_masks(D, t)
    [m for m in MASKS if all(have(D, t, m, c) for c in CHAINS)]
end

function main()
    isdir(DIR) || (println("no output directory $DIR -- nothing to summarize"); return)

    allcfg = vcat([BASELINE], CONFIGS)
    avail = Dict((D, t) => complete_masks(D, t) for (D, t, _) in allcfg)
    for (D, t, lab) in allcfg
        n = length(avail[(D,t)])
        n == length(MASKS) || @printf("NOTE: %-28s has %d/%d complete splits\n", lab, n, length(MASKS))
    end

    base_masks = avail[(BASELINE[1], BASELINE[2])]
    if isempty(base_masks)
        println("\nno complete baseline (D = 1) runs; information gain cannot be computed")
    else
        println("\n", "="^92)
        println("Held-out forecasting information gain over the D = 1 Poisson baseline")
        println("(nats per held-out county-day; positive favours the order-statistic model)")
        println("="^92)
        @printf("%-30s %14s %22s\n", "model", "info rate", "gain vs Poisson")
        println("-"^92)
        base = Dict(m => inforate_pooled(BASELINE[1], BASELINE[2], m) for m in base_masks)
        @printf("%-30s %14.4f %22s\n", BASELINE[3], mean(values(base)), "--")
        for (D, t, lab) in CONFIGS
            ms = intersect(avail[(D,t)], base_masks)
            isempty(ms) && continue
            gains = [inforate_pooled(D, t, m) - base[m] for m in ms]
            se = length(ms) > 1 ? std(gains)/sqrt(length(ms)) : NaN
            @printf("%-30s %14.4f %14.4f +/- %.4f\n",
                    lab, mean(inforate_pooled(D, t, m) for m in ms), mean(gains), 1.96*se)
        end
        println("SE is over the ", length(base_masks), " train/test splits.")
    end

    println()
    println("="^116)
    println("MCMC diagnostics and computing performance")
    println("="^116)
    @printf("%-30s %8s %9s %8s %8s %8s %8s %9s %9s\n",
            "model", "s/chain", "s/100it", "Rhat_lp", "ESSb_lp", "ESSt_lp", "Rhat_mu", "ESSb_mu", "ESS/s")
    println("-"^116)
    for (D, t, lab) in allcfg
        ms = avail[(D,t)]
        isempty(ms) && continue
        times = Float64[]; nsweeps = 0
        rh_lp = Float64[]; eb_lp = Float64[]; et_lp = Float64[]
        rh_mu = Float64[]; eb_mu = Float64[]
        for m in ms
            for c in CHAINS
                r = loadrun(D, t, m, c)
                push!(times, r["time"]); nsweeps = r["nsweeps"]
            end
            L = draws_matrix(D, t, m, "heldout_loglik_draws")   # (draws, chains)
            A3 = reshape(L, size(L,1), size(L,2), 1)
            push!(rh_lp, safe(rhat, A3)[1])
            push!(eb_lp, ess_bulk(A3)[1]); push!(et_lp, ess_tail(A3)[1])
            MU = mu_array(D, t, m)
            push!(rh_mu, maximum(safe(rhat, MU)))
            push!(eb_mu, mean(filter(isfinite, ess_bulk(MU))))
        end
        tmed = median(times)
        # ESS is produced by all four chains together, so the cost that buys it is 4 chains.
        cost4 = 4 * tmed
        @printf("%-30s %8.0f %9.2f %8.3f %8.0f %8.0f %8.3f %9.0f %9.2f\n",
                lab, tmed, 100*tmed/nsweeps,
                maximum(rh_lp), mean(eb_lp), mean(et_lp),
                maximum(rh_mu), mean(eb_mu), mean(eb_lp)/cost4)
    end

    # Environment: a timing column is not interpretable without it, and it must be the
    # same across every run in the table.
    println()
    envs = Set{Tuple}()
    for (D, t, _) in allcfg, m in avail[(D,t)], c in CHAINS
        r = loadrun(D, t, m, c)
        push!(envs, (r["cpu"], r["nthreads"], r["julia"], r["gitcommit"]))
    end
    if isempty(envs)
        println("environment: no complete runs to report")
    else
        println("environment (cpu, threads, julia, commit):")
        for e in envs; println("  ", e); end
        length(envs) == 1 || println("  WARNING: runs span more than one environment; the timing column mixes machines or code versions.")
    end
end

main()
