# Is the low-D chain's growth term inflated?
#
#   julia --project=. revision2_experiments/covid/growth_vs_d.jl <dir> [<dir> ...]
#
# The D-level mode in the K = 3 runs (one chain at mean D ~5.7-5.9, three at 7.78-7.81, its
# held-out 1300-1600 nats worse) is hypothesised to be a joint mode in (theta'phi, D) rather
# than a mode in the order layer alone: with eps FIXED at 0.5, the only thing that sets the
# gap between mu_{i,t} and the observation is the growth term, and a wider gap needs less
# concentration, i.e. a lower D. If that is right the low-D chain carries a systematically
# LARGER growth surface, and the lever is the growth scale, not D's kernel.
#
# Everything here is read from the stored functionals -- no refitting. Note the chain files'
# "mu" is theta'phi WITHOUT the Y_{i,t-1} offset and without eps (run_diag_chain.jl line
# ~199), which is exactly the growth term we want.

using JLD, Printf, Statistics

const BURN = 7500          # window 7500-15000, as in every reported table
const SEGLO = 5            # NSEG = 8 equal segments of a 15000-sweep run, so 5:8 is the window

read1(f, k) = jldopen(f, "r") do fh; read(fh, k) end

function chainrow(f)
    sw = read1(f, "sw"); dm = read1(f, "dm"); ll = read1(f, "ll")
    mu = read1(f, "mu"); dc = read1(f, "dc")
    Dmax = read1(f, "Dmax"); N = read1(f, "N"); M = read1(f, "M")
    w = findall(>=(BURN), sw)
    isempty(w) && error("$f: no draws past $BURN")

    meanD   = mean(dm[w])
    heldout = mean(ll[w])
    # tracked cells (300, identical across chains by construction: seed 20260810)
    gtrack  = mean(mean(Float64.(mu[i])) for i in w)
    gmedian = median(vcat((Float64.(mu[i]) for i in w)...))
    atmax   = mean(mean(dc[i] .== Dmax) for i in w)
    atone   = mean(mean(dc[i] .== 1)    for i in w)

    # full surface, from the per-segment sufficient statistics
    S1mu = read1(f, "S1mu"); S1d = read1(f, "S1d"); segn = read1(f, "segn")
    nseg = size(S1mu, 3)
    ndr  = sum(segn[SEGLO:nseg])
    gfull = sum(Float64, @view S1mu[:, :, SEGLO:nseg]) / (ndr * N * M)
    dfull = sum(Float64, @view S1d[:, :, SEGLO:nseg])  / (ndr * N * M)
    # per-cell posterior mean D over the window; ">= Dmax - 0.5" = effectively pinned
    cellD = dropdims(sum(Float64, @view S1d[:, :, SEGLO:nseg]; dims=3); dims=3) ./ ndr
    fracpin = mean(cellD .>= Dmax - 0.5)

    (; meanD, dfull, heldout, gtrack, gmedian, gfull, atmax, atone, fracpin, ndraws=length(w))
end

for dir in ARGS
    println("\n=== ", dir, " ===")
    files = sort(filter(f -> occursin(r"diag_chain\d+\.jld$", f), readdir(dir; join=true)))
    isempty(files) && (println("  no chain files"); continue)
    rows = Tuple{String,NamedTuple}[]
    for f in files
        try
            push!(rows, (basename(f), chainrow(f)))
        catch e
            println("  ", basename(f), ": ", e)
        end
    end
    @printf("%-18s %7s %7s %10s %12s %12s %7s %7s %7s\n",
            "chain", "meanD", "D(full)", "held-out", "growth(trk)", "growth(full)",
            "@Dmax", "@D=1", "pinned")
    for (nm, r) in rows
        @printf("%-18s %7.3f %7.3f %10.1f %12.1f %12.1f %6.1f%% %6.1f%% %6.1f%%\n",
                nm, r.meanD, r.dfull, r.heldout, r.gtrack, r.gfull,
                100r.atmax, 100r.atone, 100r.fracpin)
    end
    if length(rows) >= 2
        ds = [r.meanD for (_, r) in rows]; gs = [r.gfull for (_, r) in rows]
        lo = argmin(ds); hi = setdiff(1:length(ds), [lo])
        @printf("  lowest-D chain %s: meanD %.3f vs %.3f (others), growth %.1f vs %.1f  => %+.1f%%\n",
                rows[lo][1], ds[lo], mean(ds[hi]), gs[lo], mean(gs[hi]),
                100 * (gs[lo] / mean(gs[hi]) - 1))
        @printf("  rank corr(meanD, growth) across chains: %s\n",
                length(ds) >= 3 ? string(round(cor(ds, gs), digits=3)) : "n/a")
    end
end
