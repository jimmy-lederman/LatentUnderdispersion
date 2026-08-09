# Verify the flights campaign before trusting any of it.
#   1. coverage   -- which of the expected runs exist, which are missing
#   2. fairness   -- did every TIMED run land on dedicated, identical hardware?
#   3. preview    -- per-config wall-clock from the dedicated runs only
#
# usage: julia --project=. revision2_experiments/flights/cluster/check_runs.jl
using JLD, Printf, Statistics

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DIR  = joinpath(ROOT, "output/flights/revisionsamples")

const CONFIGS = [(0,1,0),(1,1,0),(3,1,0),(5,1,0),(7,1,0),(9,1,0),(0,2,2),(0,3,0),(1,3,0),(0,4,2)]
const MASKS   = [101,102,103,104,105]
const CHAINS  = [1,2,3,4]

label(D,t,g) = t == 1 ? "MedPois hier D=$D"  : t == 2 ? "STAR hier g=$g" :
               t == 3 ? "MedPois flat D=$D"  : "STAR flat g=$g"
fname(D,t,g,m,c) = t == 1 ? "MedPoissonD$(D)mask$(m)chain$(c).jld" :
                   t == 2 ? "STARg$(g)mask$(m)chain$(c).jld" :
                   t == 3 ? "MedPoissonFlatD$(D)mask$(m)chain$(c).jld" :
                            "STARFlatg$(g)mask$(m)chain$(c).jld"

function coverage()
    println("="^78); println("1. COVERAGE")
    miss = String[]; found = 0
    for (D,t,g) in CONFIGS, m in MASKS, c in CHAINS
        f = joinpath(DIR, fname(D,t,g,m,c))
        (isfile(f) && filesize(f) > 0) ? (found += 1) : push!(miss, fname(D,t,g,m,c))
    end
    @printf("   %d / %d runs present\n", found, length(CONFIGS)*length(MASKS)*length(CHAINS))
    if !isempty(miss)
        @printf("   MISSING (%d):\n", length(miss))
        for f in first(miss, 12); println("     ", f); end
        length(miss) > 12 && println("     ... and $(length(miss)-12) more")
    end
end

function collect_dedicated()
    ded = Dict{Tuple{Int,Int,Int},Vector{Any}}()
    hosts = String[]; cpus = String[]; cores = String[]; commits = String[]; threads = String[]
    for (D,t,g) in CONFIGS, m in MASKS, c in CHAINS
        f = joinpath(DIR, fname(D,t,g,m,c))
        (isfile(f) && filesize(f) > 0) || continue
        d = load(f)
        get(d, "dedicated", false) || continue
        push!(get!(ded, (D,t,g), Any[]), d)
        push!(hosts, string(get(d,"hostname","?")))
        push!(cpus, string(get(d,"cpu","?")))
        push!(cores, string(get(d,"ncores",0)))
        push!(commits, string(get(d,"gitcommit","?")))
        push!(threads, string(get(d,"nthreads",0)))
    end
    ded, hosts, cpus, cores, commits, threads
end

function fairness(ded, hosts, cpus, cores, commits, threads)
    println("\n2. TIMING-RUN FAIRNESS  (only runs saved with FLIGHTS_DEDICATED=1 are usable)")
    if isempty(hosts)
        println("   no dedicated runs found -- submit_timing.sbatch has not completed")
        return
    end
    @printf("   %d dedicated runs across %d configs\n", length(hosts), length(ded))
    ok = true
    for (nm, vals) in (("CPU model", cpus), ("core count", cores),
                       ("thread count", threads), ("git commit", commits))
        u = unique(vals)
        if length(u) == 1
            @printf("   OK    %-12s identical everywhere: %s\n", nm, u[1])
        else
            ok = false
            @printf("   FAIL  %-12s DIFFERS across runs: %s\n", nm, join(u, ", "))
            println("         -> timings are NOT comparable; rerun on one node class")
        end
    end
    # Node reuse across array tasks is fine: --exclusive guarantees a run had the node to
    # itself while it ran, and a later task may be placed on the same node afterwards.
    # A hostname cannot distinguish sequential reuse from concurrency, so this is
    # reported as information only -- exclusivity is enforced by SLURM, not checked here.
    @printf("   info  %d dedicated run(s) over %d distinct node(s): %s\n",
            length(hosts), length(unique(hosts)), join(sort(unique(hosts)), ", "))
    for cfg in CONFIGS
        haskey(ded, cfg) || @printf("   WARN  no dedicated timing run for %s\n", label(cfg...))
    end
    ok && println("   => hardware is homogeneous; wall-clock is comparable across models")
end

function preview(ded)
    isempty(ded) && return
    println("\n3. TIMING PREVIEW  (dedicated runs only; median over replicates)")
    @printf("   %-22s %5s %10s %12s %12s\n", "model", "n", "s/chain", "s/100 iters", "inforate")
    for cfg in CONFIGS
        haskey(ded, cfg) || continue
        v = ded[cfg]
        ts = [x["time"] for x in v]
        @printf("   %-22s %5d %10.2f %12.3f %12.5f\n", label(cfg...), length(v),
                median(ts), 100*median(ts)/v[1]["nsweeps"], mean([x["inforate"] for x in v]))
    end
    println("\n   (information gain = each model's inforate minus the D=1 baseline's)")
end

function main()
    isdir(DIR) || error("no results directory at $DIR")
    coverage()
    ded, hosts, cpus, cores, commits, threads = collect_dedicated()
    fairness(ded, hosts, cpus, cores, commits, threads)
    preview(ded)
    println("="^78)
end

main()
