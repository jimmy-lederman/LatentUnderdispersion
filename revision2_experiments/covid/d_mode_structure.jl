# What distinguishes the low-D chain? Structure of the D deficit.
#
#   julia --project=. revision2_experiments/covid/d_mode_structure.jl <dir> [<dir> ...]
#
# growth_vs_d.jl established that the low-D chain's growth surface matches its siblings to
# within 2%, so every cell's LIKELIHOOD for D is essentially identical across chains and the
# difference must enter through the OddBinomial prior, rho_{i,t} = logit^{-1}(beta_i tau_t).
# beta and tau are not stored, but at Q = 1 that linear predictor is RANK ONE, so a mode in
# the rho layer must leave a rank-one footprint in the D surface itself. That is testable
# from the stored per-segment sufficient statistics.
#
# Three things are computed per run:
#
#   1. mean D against sweep, in blocks, so a chain that never climbed can be told apart from
#      one that climbed and fell back (and, for the annealed run, what happened at release).
#   2. SVD of the per-cell D deficit (good chains' mean minus the low chain), with the
#      deficit between two GOOD chains as the null. A rank-one rho mode predicts the first
#      singular value carries most of the signal variance while the null does not.
#   3. Where the deficit sits: against the observed increment (the known driver of D, see
#      d_by_increment.jl) and against the growth surface.

using JLD, Printf, Statistics, LinearAlgebra, CSV, DataFrames

const BURN  = 7500
const SEGLO = 5          # NSEG = 8, so segments 5:8 are the 7500-15000 window
const ROOTDIR = abspath(joinpath(@__DIR__, "../.."))

read1(f, k) = jldopen(f, "r") do fh; read(fh, k) end

# per-cell posterior mean over the window, from the segment sums
function cellmeans(f)
    S1d = read1(f, "S1d"); S1mu = read1(f, "S1mu"); segn = read1(f, "segn")
    nseg = size(S1d, 3); ndr = sum(segn[SEGLO:nseg])
    D = dropdims(sum(Float64, @view S1d[:, :, SEGLO:nseg]; dims=3); dims=3) ./ ndr
    G = dropdims(sum(Float64, @view S1mu[:, :, SEGLO:nseg]; dims=3); dims=3) ./ ndr
    (D, G)
end

function svdreport(label, Δ)
    s = svd(Δ).S
    tot = sum(abs2, s)
    @printf("    %-26s ||Δ||_F/cell %7.4f   var in rank 1/2/3: %5.1f%% %5.1f%% %5.1f%%\n",
            label, sqrt(tot / length(Δ)),
            100s[1]^2/tot, 100(s[1]^2+s[2]^2)/tot, 100(s[1]^2+s[2]^2+s[3]^2)/tot)
    return s
end

# ---- increments, for the "where does the deficit sit" cross-tab -------------------------
cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
Y0 = Int.(cumdf[:, 1]); Yd = cumdf[:, 2:end]
N, M = size(Yd)
inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i, 1] = max(Yd[i, 1] - Y0[i], 0)
    for t in 2:M; inc[i, t] = max(Yd[i, t] - Yd[i, t-1], 0); end
end
println("data: N = $N, M = $M")

for dir in ARGS
    println("\n", "="^92, "\n=== ", dir)
    files = sort(filter(f -> occursin(r"diag_chain\d+\.jld$", f), readdir(dir; join=true)))
    length(files) < 2 && (println("  need >= 2 chains"); continue)

    # ---- 1. mean D against sweep -------------------------------------------------------
    println("\n  mean D by sweep block:")
    @printf("    %-8s", "block")
    for f in files; @printf(" %8s", replace(basename(f), "diag_chain" => "c", ".jld" => "")); end
    println()
    sws = [read1(f, "sw") for f in files]; dms = [read1(f, "dm") for f in files]
    for b in 0:1000:14000
        @printf("    %-8s", "$(b+1)-$(b+1000)")
        for (sw, dm) in zip(sws, dms)
            w = findall(s -> b < s <= b + 1000, sw)
            isempty(w) ? @printf(" %8s", "-") : @printf(" %8.3f", mean(dm[w]))
        end
        println()
    end

    # ---- per-cell window means ---------------------------------------------------------
    Ds = Matrix{Float64}[]; Gs = Matrix{Float64}[]
    for f in files; (D, G) = cellmeans(f); push!(Ds, D); push!(Gs, G); end
    meanD = [mean(D) for D in Ds]
    lo = argmin(meanD); good = setdiff(1:length(Ds), [lo])
    @printf("\n  low chain = c%d (mean D %.3f); good = %s (%.3f)\n",
            lo, meanD[lo], join(["c$g" for g in good], ","), mean(meanD[good]))
    length(good) < 2 && (println("  need >= 2 good chains for the null"); continue)

    # ---- 2. SVD of the deficit, against a good-vs-good null -----------------------------
    Dgood = sum(Ds[g] for g in good) ./ length(good)
    println("\n  SVD of the per-cell D difference:")
    svdreport("good(mean) - low", Dgood .- Ds[lo])
    for a in 1:length(good)-1, b in a+1:length(good)
        svdreport("NULL c$(good[a]) - c$(good[b])", Ds[good[a]] .- Ds[good[b]])
    end

    # rank-1 reconstruction of the deficit: is it separable in (county, day)?
    Δ = Dgood .- Ds[lo]
    F = svd(Δ)
    u = F.U[:, 1] .* F.S[1]; v = F.Vt[1, :]
    @printf("    rank-1 fit: corr(Δ, u1*v1') = %.4f;  county loading |u| spread p10/p50/p90 = %.3f/%.3f/%.3f\n",
            cor(vec(Δ), vec(u * v')),
            quantile(abs.(u), 0.1), quantile(abs.(u), 0.5), quantile(abs.(u), 0.9))

    # ---- 3. where the deficit sits -----------------------------------------------------
    println("\n  deficit by observed increment:")
    buckets = [(-1, 0, "inc == 0"), (0, 1, "inc == 1"), (1, 5, "inc 2-5"),
               (5, 20, "inc 6-20"), (20, 100, "inc 21-100"), (100, typemax(Int), "inc > 100")]
    @printf("    %-12s %8s %10s %10s %10s\n", "bucket", "share", "D low", "D good", "deficit")
    for (a, b, nm) in buckets
        m = (inc .> a) .& (inc .<= b)
        s = count(m); s == 0 && continue
        @printf("    %-12s %7.1f%% %10.3f %10.3f %10.3f\n",
                nm, 100s/length(inc), mean(Ds[lo][m]), mean(Dgood[m]), mean(Δ[m]))
    end
    @printf("\n  corr(deficit, log1p(increment))      = %+.4f\n", cor(vec(Δ), vec(log1p.(inc))))
    @printf("  corr(deficit, growth of good chains) = %+.4f\n",
            cor(vec(Δ), vec(sum(Gs[g] for g in good) ./ length(good))))
    # county vs day concentration
    rm_ = vec(mean(Δ, dims=2)); cm_ = vec(mean(Δ, dims=1))
    @printf("  deficit sd: overall %.4f | across counties (row means) %.4f | across days (col means) %.4f\n",
            std(vec(Δ)), std(rm_), std(cm_))
end
