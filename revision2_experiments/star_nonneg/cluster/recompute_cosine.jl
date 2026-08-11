# Factor-recovery under the CORRECT invariance group for each prior, from the
# saved posterior samples -- no refitting.
#
# Referee 2 (point 3) objects that latent factors are identified only up to
# permutation and sign, and for Gaussian priors up to ROTATION. The sweep's
# cosine metric (cell_lib.jl) aligns by permutation and sign only. That is the
# right invariance group for the non-negative models -- non-negativity rules out
# rotation, since a rotated non-negative factor generally leaves the positive
# orthant -- but it is NOT the right group for STAR under iid Gaussian priors on
# U and V, where the likelihood depends on UV and the prior is orthogonally
# invariant, so U is identified only up to U -> UR for orthogonal R.
#
# Scoring a rotation-invariant posterior with a permutation-only metric can
# report failure when the subspace was in fact recovered. On CMP star_id seed 1
# the shipped metric gives 0.273 while Procrustes alignment gives 0.923.
#
# This computes BOTH, for every cell, so the paper can report the comparison
# under each model's own invariance group and say which is which:
#
#   cos_perm  -- permutation + sign, replicating cell_lib.jl exactly (note the
#                argument order of best_perm: S = cor(reference, candidate))
#   cos_proc  -- optimal orthogonal rotation (Procrustes) of each draw onto the
#                truth, averaged over draws, then column-wise cosine
#
# Procrustes is the more generous metric (K^2 continuous parameters versus K!
# discrete choices), so cos_proc >= cos_perm essentially always; the informative
# quantity is the GAP, which measures how much of the mismatch was pure rotation.
# For a non-negative model the gap should be near zero -- that is the internal
# check that the implementation is sound, and it held on two test cells (medpois
# 0.942 vs 0.942, mednb 0.989 vs 0.991).
#
# Usage: julia --project=../../.. recompute_cosine.jl <task_id> <njobs>
# Writes results_cos/*.csv; completed cells are skipped.

using JLD, CSV, DataFrames, LinearAlgebra, Statistics, Combinatorics, Printf

task_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
njobs   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

const SAMPDIR = joinpath(@__DIR__, "results", "samples")
const OUTDIR  = joinpath(@__DIR__, "results_cos")
const DATADIR = joinpath(@__DIR__, "../../../revison_experiments/factorexperiment/data")
const TRUTH = Dict("CMP" => "CMPU_NK2.csv", "GC" => "GCU_NK.csv",
                   "Poisson" => "PoisU_NK.csv", "NB" => "NBU_NK.csv")
mkpath(OUTDIR)

function loadtruth(ds, N)
    A = Matrix(CSV.read(joinpath(DATADIR, TRUTH[ds]), DataFrame))[1:end, 2:end]
    return size(A, 1) == N ? Float64.(A) : Float64.(permutedims(A))
end

# exactly cell_lib.jl's best_perm(Ur, Ut): S = abs.(cor(Ur, Ut)), pick p
# maximizing sum(S[k, p[k]]), then the caller does Ut[:, p].
function best_perm(Ur, Ut, K)
    S = abs.(cor(Ur, Ut))
    best = -Inf; bp = collect(1:K)
    for p in permutations(1:K)
        sc = sum(S[k, p[k]] for k in 1:K)
        if sc > best; best = sc; bp = collect(p); end
    end
    return bp
end

colcos(Ut, Um, K) = mean(abs(dot(Ut[:, k], Um[:, k])) /
                         (norm(Ut[:, k]) * norm(Um[:, k])) for k in 1:K)

function cosines(path, ds)
    d = JLD.load(path)
    U = d["U"]; nch, ns = d["nchains"], d["nsamples"]
    N, K = size(U, 2), size(U, 3)
    Ut = loadtruth(ds, N)
    G = nch * ns

    # --- shipped metric: per-chain permutation to that chain's first draw,
    # chains aligned to chain 1, posterior mean, final permutation to truth ---
    Ua = Array{Float64}(undef, G, N, K)
    for ch in 1:nch
        refc = Float64.(@view U[(ch - 1) * ns + 1, :, :])
        for s in 1:ns
            g = (ch - 1) * ns + s
            Us = Float64.(@view U[g, :, :])
            Ua[g, :, :] = Us[:, best_perm(refc, Us, K)]
        end
    end
    ref = Ua[1, :, :]
    for ch in 2:nch
        lo = (ch - 1) * ns + 1; hi = ch * ns
        p = best_perm(ref, dropdims(mean(Ua[lo:hi, :, :], dims = 1), dims = 1), K)
        for s in lo:hi
            Ua[s, :, :] = Ua[s, :, :][:, p]
        end
    end
    Um = dropdims(mean(Ua, dims = 1), dims = 1)
    Um = Um[:, best_perm(Ut, Um, K)]
    cperm = colcos(Ut, Um, K)

    # --- Procrustes: rotate each draw onto the truth, then average ---
    acc = zeros(N, K)
    for g in 1:G
        Us = Float64.(@view U[g, :, :])
        F = svd(Us' * Ut)
        acc .+= Us * (F.U * F.Vt)
    end
    acc ./= G
    cproc = colcos(Ut, acc, K)

    return (cos_perm = cperm, cos_proc = cproc)
end

files = sort(filter(f -> endswith(f, "_samples.jld"), readdir(SAMPDIR)))
mine = files[task_id:njobs:length(files)]
println("task $task_id/$njobs: $(length(mine)) cells")
flush(stdout)

for (i, f) in enumerate(mine)
    stem = replace(f, "_samples.jld" => "")
    out = joinpath(OUTDIR, stem * ".csv")
    (isfile(out) && filesize(out) > 0) && continue
    parts = split(stem, "_")
    ds = parts[1]; seed = parse(Int, parts[end])
    mname = join(parts[2:end-1], "_")
    haskey(TRUTH, ds) || continue
    try
        r = cosines(joinpath(SAMPDIR, f), ds)
        CSV.write(out, DataFrame(; dataset = ds, model = mname, seed = seed, pairs(r)...))
    catch e
        println("FAILED $stem: ", first(sprint(showerror, e), 300))
    end
    i % 25 == 0 && (println("  $i/$(length(mine))"); flush(stdout))
end
println("task $task_id complete")
