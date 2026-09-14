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

function align_draws(thw, niter=8)
    perms = collect(permutations(1:K))
    ref = mean(thw[1]); nc = length(thw)
    asg = [[collect(1:K) for _ in eachindex(thw[c])] for c in 1:nc]
    for _ in 1:niter
        acc = zeros(size(ref)); n = 0
        for c in 1:nc, s in eachindex(thw[c])
            p = perms[argmin([sum(abs2, ref .- thw[c][s][:, pp]) for pp in perms])]
            asg[c][s] = p; acc .+= thw[c][s][:, p]; n += 1
        end
        ref = acc ./ n
    end
    return asg
end

hi = sw[end]
windows = [(0, hi ÷ 4), (hi ÷ 4, hi ÷ 2), (hi ÷ 2, hi)]
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

    thw = [[Float64.(ch[c]["th"][i]) for i in idx] for c in 1:nchains]
    phw = [[Float64.(ch[c]["ph"][i]) for i in idx] for c in 1:nchains]
    asg = align_draws(thw)
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
                    ("theta/s_k aligned", TH), ("s_k*phi aligned", PH),
                    ("FACTOR centre of mass", COM), ("FACTOR activity share", SHR))
        (r, ec, ep) = stats(A)
        @printf("%-14s %-24s %6.2f/%6.2f %7.0f/%6.0f/%6.0f\n", w, nm, r[1], r[2], ec[1], ec[2], ep)
    end
    println()
end
