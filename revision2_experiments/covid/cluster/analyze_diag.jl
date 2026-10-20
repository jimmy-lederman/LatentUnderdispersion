# Combine the per-chain diagnostic runs and report the convergence table.
#
#   julia --project=. analyze_diag.jl [outdir]
#
# Reports, for each quantity and each run-length window:
#   Rhat  median / max      (median is the headline; max is what the Vehtari et al. /
#                            Stan convention asks for, since the claim is "all parameters",
#                            so both are shown)
#   ESS   median / min / pooled
#         per-chain bulk ESS summarised over (chain, parameter), plus the POOLED ESS across
#         chains. When Rhat > 1 the pooled figure is much worse than the sum of the
#         per-chain ones, because between-chain disagreement registers as variance -- so
#         pooled is the honest headline and per-chain is the optimistic one.
#
# Factor labels are relabelled PER DRAW against an iterated consensus reference. Matching
# chain MEANS is wrong: label switching within a chain smears the mean and makes the match
# meaningless.

using JLD, Statistics, Printf, LinearAlgebra, Combinatorics
using MCMCDiagnosticTools

ROOTDIR = abspath(joinpath(@__DIR__, "../../.."))
outdir  = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOTDIR, "output/covid/diag")

files = sort(filter(f -> startswith(f, "diag_chain") && endswith(f, ".jld"), readdir(outdir)))
isempty(files) && (println("no diag_chain*.jld in $outdir"); exit(1))
ch = [load(joinpath(outdir, f)) for f in files]
nchains = length(ch)
K = ch[1]["K"]; M = ch[1]["M"]
println("chains: ", join([c["chainSeed"] for c in ch], ", "))
@printf("N = %d, M = %d, K = %d, Q = %d, Dmax = %d, eps fixed at %.2f\n",
        ch[1]["N"], M, K, ch[1]["Q"], ch[1]["Dmax"], ch[1]["eps_fixed"])
for c in ch
    @printf("  chain %d: %d sweeps, %.1f min, %s, %d threads\n",
            c["chainSeed"], c["nsweeps"], c["seconds"]/60, c["cpu"], c["nthreads"])
end

nk_all = minimum(length(c["sw"]) for c in ch)
sw = ch[1]["sw"][1:nk_all]

safe(f, x) = try f(x) catch; NaN end
r3(A) = reshape(A, size(A,1), size(A,2), 1)

# per-parameter Rhat, per-(chain,parameter) single-chain ESS, and per-parameter pooled ESS
function stats(A)
    P = size(A, 3)
    rs  = filter(isfinite, [safe(_ -> rhat(r3(A[:,:,p]))[1], p) for p in 1:P])
    esp = filter(isfinite, [safe(_ -> ess(r3(A[:,:,p]); kind=:bulk)[1], p) for p in 1:P])
    esc = filter(isfinite, [safe(x -> ess(r3(reshape(x, :, 1)); kind=:bulk)[1], A[:,c,p])
                            for c in 1:size(A,2), p in 1:P])
    return (isempty(rs) ? (NaN,NaN) : (median(rs), maximum(rs)),
            isempty(esc) ? (NaN,NaN) : (median(esc), minimum(esc)),
            isempty(esp) ? NaN : median(esp))
end

function align_draws(thw, phw=nothing, niter=8)   # thw[c][s] :: nI x K ; phw[c][s] :: K x M
    # Alignment by the TRAJECTORIES phi (K x M, 295 dims) when available: the 60-county
    # loading profiles are ambiguous between similar factors (Q = 1 run: chain 3 flipped among
    # three permutations in 50% of draws under theta-alignment, 0% under phi-alignment), and
    # a wrong permutation inflates Rhat of both theta and phi. Labels do not switch within a
    # Gibbs chain, so the per-draw permutation is also reported as a per-chain modal one.
    perms = collect(permutations(1:K))
    use = phw === nothing ? thw : phw
    pick(ref, x, p) = phw === nothing ? x[:, p] : x[p, :]
    ref = mean(use[1])
    assign = [[collect(1:K) for _ in eachindex(use[c])] for c in 1:nchains]
    for _ in 1:niter
        acc = zeros(size(ref)); n = 0
        for c in 1:nchains, s in eachindex(use[c])
            p = perms[argmin([sum(abs2, ref .- pick(ref, use[c][s], pp)) for pp in perms])]
            assign[c][s] = p
            acc .+= pick(ref, use[c][s], p); n += 1
        end
        ref = acc ./ n
    end
    return assign
end
# one permutation per chain (the modal one), for statistics accumulated over all counties
chain_perm(asg) = [asg[c][argmax([count(==(p), asg[c]) for p in asg[c]])] for c in eachindex(asg)]

hi = sw[end]
windows = [(0, hi ÷ 4), (hi ÷ 4, hi ÷ 2), (hi ÷ 2, hi)]

# ---- FULL-SURFACE split-Rhat (every N*M cell) from the per-segment sufficient statistics.
# A window is a set of whole segments; each chain's window is split into its first and
# second half (by segments), giving 2*nchains half-chains of n draws each. Classic
# split-Rhat: W = mean within-half variance, B = n * var(half means),
# Rhat = sqrt(((n-1)/n * W + B/n) / W). Cells with W = 0 (constant in every half) are
# excluded and counted; for D they are reported as constant-and-identical when all halves
# share the value.
function surface_rhat(S1, S2, segn, segs)
    nh = length(segs) ÷ 2
    halves = [segs[1:nh], segs[nh+1:end]]
    means = Vector{Matrix{Float64}}(); vars = Vector{Matrix{Float64}}(); n = 0
    for c in 1:nchains, h in halves
        s1 = sum(Float64.(S1[c][:, :, k]) for k in h); s2 = sum(Float64.(S2[c][:, :, k]) for k in h)
        n = sum(segn[c][k] for k in h)
        m = s1 ./ n; v = max.(s2 ./ n .- m .^ 2, 0.0) .* (n / max(n - 1, 1))
        push!(means, m); push!(vars, v)
    end
    W = mean(vars); Mbar = mean(means)
    B = n .* mean([(m .- Mbar) .^ 2 for m in means]) .* (length(means) / max(length(means) - 1, 1))
    R = sqrt.(((n - 1) / n .* W .+ B ./ n) ./ W)
    const_all = [all(vars[j][i] == 0 for j in eachindex(vars)) for i in eachindex(W)]
    ident = [const_all[i] && all(means[j][i] == means[1][i] for j in eachindex(means)) for i in eachindex(W)]
    (R = R, ok = .!const_all, ident = ident)
end
hasSurf = all(haskey(ch[c], "S1mu") for c in 1:nchains)
if hasSurf
    nseg = ch[1]["nseg"]; nsw = ch[1]["nsweeps"]
    println("\nFULL-SURFACE split-Rhat over all N*M = $(ch[1]["N"]*ch[1]["M"]) cells (from per-segment sufficient statistics)")
    @printf("%-14s %-16s %8s %8s %8s %8s   %s\n", "window", "quantity", "median", "p90", "p99", "max", "cells excluded (constant)")
    for (lo, up) in windows
        segs = [k for k in 1:nseg if lo < (k - 0.5) * nsw / nseg <= up]   # windows are in sweeps
        length(segs) < 2 && continue
        S1mu = [ch[c]["S1mu"] for c in 1:nchains]; S2mu = [ch[c]["S2mu"] for c in 1:nchains]
        S1d = [ch[c]["S1d"] for c in 1:nchains]; S2d = [ch[c]["S2d"] for c in 1:nchains]; segn = [ch[c]["segn"] for c in 1:nchains]
        hasTh = all(haskey(ch[c], "S1th") for c in 1:nchains)
        if hasTh
            # permute each chain's columns by its modal permutation from the phi alignment
            idxw = findall(s -> lo < s <= up, sw); asgw = align_draws([[ch[c]["th"][i] for i in idxw] for c in 1:nchains],
                                                                      [[ch[c]["ph"][i] for i in idxw] for c in 1:nchains])
            cpw = chain_perm(asgw)
            S1th = [ch[c]["S1th"][:, cpw[c], :] for c in 1:nchains]; S2th = [ch[c]["S2th"][:, cpw[c], :] for c in 1:nchains]
        end
        for (nm, A, Bq) in (("mu (all cells)", S1mu, S2mu), ("D (all cells)", S1d, S2d),
                            (hasTh ? ("theta/s_k (all)", S1th, S2th) : ("D (all cells)", S1d, S2d)))
            r = surface_rhat(A, Bq, segn, segs); v = filter(isfinite, r.R[r.ok])
            # every cell constant (e.g. D during a D = 1 annealed warm-up): nothing to report
            if isempty(v)
                @printf("%-14s %-16s %8s %8s %8s %8s   %d of %d constant (no variation)\n", "$lo-$up", nm, "-", "-", "-", "-", count(.!r.ok), length(r.ok))
                continue
            end
            @printf("%-14s %-16s %8.3f %8.3f %8.3f %8.2f   %d of %d%s\n", "$lo-$up", nm, median(v), quantile(v, .9), quantile(v, .99), maximum(v),
                    count(.!r.ok), length(r.ok), nm[1] == 'D' ? @sprintf(" (%d identical across all chains)", count(r.ident)) : "")
        end
    end
end
println()
@printf("%-14s %-24s %13s %21s\n", "window", "quantity", "Rhat med/max", "ESS med/min/pooled")
println("-"^78)
for (lo, up) in windows
    idx = findall(s -> lo < s <= up, sw)
    length(idx) < 30 && continue
    nk = length(idx)

    L = Array{Float64}(undef, nk, nchains, 1)
    for c in 1:nchains; L[:,c,1] = ch[c]["ll"][idx]; end
    DM = Array{Float64}(undef, nk, nchains, 1)
    for c in 1:nchains; DM[:,c,1] = ch[c]["dm"][idx]; end
    P = length(ch[1]["cells"])
    MU = Array{Float64}(undef, nk, nchains, P)
    for c in 1:nchains, s in 1:nk, p in 1:P; MU[s,c,p] = ch[c]["mu"][idx[s]][p]; end
    # D at the same cells (files written after 2026-09-16 only). D is discrete on
    # {1,3,...,Dmax} and most cells sit at the ceiling in nearly every draw, where rank-Rhat
    # is undefined; Rhat is computed over the cells whose D varies, and the constant cells
    # are reported as the fraction on which all chains agree.
    hasD = all(haskey(ch[c], "dc") && length(ch[c]["dc"]) >= maximum(idx) for c in 1:nchains)
    if hasD
        DC = Array{Float64}(undef, nk, nchains, P)
        for c in 1:nchains, s in 1:nk, p in 1:P; DC[s,c,p] = ch[c]["dc"][idx[s]][p]; end
        varying = [length(unique(vec(DC[:,:,p]))) > 1 for p in 1:P]
        allsame = [all(DC[:,:,p] .== DC[1,1,p]) for p in 1:P]
        DCv = DC[:, :, varying]
        @printf("%-14s %-24s   D at cells: %d of %d vary; %d constant and identical across chains\n",
                "$lo-$up", "", count(varying), P, count(allsame))
    end

    thw = [[Float64.(ch[c]["th"][i]) for i in idx] for c in 1:nchains]
    phw = [[Float64.(ch[c]["ph"][i]) for i in idx] for c in 1:nchains]
    asg = align_draws(thw, phw)
    cp = chain_perm(asg)
    @printf("%-14s %-24s   alignment by phi: modal permutation held in %s%% of draws per chain\n", "$lo-$up", "",
            join([@sprintf("%.0f", 100 * count(==(cp[c]), asg[c]) / length(asg[c])) for c in 1:nchains], "/"))
    nI = size(thw[1][1], 1)
    TH = Array{Float64}(undef, nk, nchains, nI*K)
    PH = Array{Float64}(undef, nk, nchains, K*M)
    COM = Array{Float64}(undef, nk, nchains, K)
    SHR = Array{Float64}(undef, nk, nchains, K)
    for c in 1:nchains, s in 1:nk
        p = asg[c][s]
        TH[s,c,:] = vec(thw[c][s][:, p])
        Pk = phw[c][s][p, :]
        PH[s,c,:] = vec(Pk)
        tot = vec(sum(Pk, dims=2))
        for k in 1:K
            COM[s,c,k] = sum(t * Pk[k,t] for t in 1:M) / max(tot[k], 1e-300)
        end
        SHR[s,c,:] = tot ./ max(sum(tot), 1e-300)
    end

    w = "$lo-$up"
    for (nm, A) in (("held-out log density", L), ("mean D", DM), ("mu (cells)", MU),
                    (hasD ? ("D (varying cells)", DCv) : ("mean D", DM)),
                    ("theta/s_k aligned", TH), ("s_k*phi aligned", PH),
                    ("FACTOR centre of mass", COM), ("FACTOR activity share", SHR))
        (r, ec, ep) = stats(A)
        @printf("%-14s %-24s %6.2f/%6.2f %7.0f/%6.0f/%6.0f\n", w, nm, r[1], r[2], ec[1], ec[2], ep)
    end
    println()
end
