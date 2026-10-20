# How do the four full-matrix chains differ? Per-chain summaries over the last half of the
# run: held-out density, mean D, growth surface, and the aligned factor profiles.
#   julia --project=. revision2_experiments/covid/mode_census.jl output/covid/diag_hybrid
using JLD, Statistics, Printf, LinearAlgebra, Combinatorics
dir = ARGS[1]
ch = [load(joinpath(dir, "diag_chain$c.jld")) for c in 1:4]
K = ch[1]["K"]; M = ch[1]["M"]
half(x) = x[(length(x) ÷ 2 + 1):end]
println("dir = $dir, last half of ", ch[1]["nsweeps"], " sweeps (thin ", ch[1]["thin"], ")\n")
# per-chain scalars
@printf("%-6s %14s %10s %12s\n", "chain", "held-out ll", "mean D", "mean growth")
for c in 1:4
    ll = half(ch[c]["ll"]); dm = half(ch[c]["dm"]); mu = half(ch[c]["mu"])
    @printf("%-6d %14.1f %10.3f %12.2f\n", c, mean(ll), mean(dm), mean(mean.(mu)))
end
# growth surface at the 150 tracked cells: per-chain posterior mean; pairwise agreement
MU = [mean(reduce(hcat, half(ch[c]["mu"])), dims=2)[:] for c in 1:4]
println("\nlog growth at 150 cells, pairwise correlation / RMS log-ratio between chain means:")
for i in 1:4, j in i+1:4
    a = log.(MU[i] .+ 1e-3); b = log.(MU[j] .+ 1e-3)
    @printf("  chains %d-%d: corr %.3f   rms |log ratio| %.2f\n", i, j, cor(a, b), sqrt(mean((a .- b).^2)))
end
# aligned factor profiles: per-chain mean of s_k*phi (K x M), aligned to chain 1 by the best permutation
P = [mean(half(ch[c]["ph"])) for c in 1:4]                    # K x M each
perms = collect(permutations(1:K))
cosr(a, b) = dot(a, b) / (norm(a) * norm(b))
println("\nfactor profiles s_k*phi (chain means), aligned to chain 1: per-factor cosine similarity, centre of mass (days)")
com(v) = sum(t * v[t] for t in 1:M) / sum(v)
@printf("  chain 1 centres: %s   shares: %s\n", join([@sprintf("%.0f", com(P[1][k,:])) for k in 1:K], " "),
        join([@sprintf("%.2f", sum(P[1][k,:]) / sum(P[1])) for k in 1:K], " "))
for c in 2:4
    best = perms[argmax([sum(cosr(P[1][k,:], P[c][p[k],:]) for k in 1:K) for p in perms])]
    cs = [cosr(P[1][k,:], P[c][best[k],:]) for k in 1:K]
    @printf("  chain %d: cos %s   centres: %s   shares: %s\n", c, join([@sprintf("%.2f", x) for x in cs], " "),
            join([@sprintf("%.0f", com(P[c][best[k],:])) for k in 1:K], " "),
            join([@sprintf("%.2f", sum(P[c][best[k],:]) / sum(P[c])) for k in 1:K], " "))
end
# within-chain spread vs between-chain spread of the total growth trajectory sum_k s_k phi_k(t)
T = [reduce(hcat, [vec(sum(p, dims=1)) for p in half(ch[c]["ph"])]) for c in 1:4]   # M x draws
wsd = mean([mean(std(T[c], dims=2)) for c in 1:4]); bsd = mean(std(reduce(hcat, [mean(T[c], dims=2) for c in 1:4]), dims=2))
@printf("\ntotal growth trajectory sum_k s_k phi_k(t): within-chain sd %.1f, between-chain sd of means %.1f\n", wsd, bsd)
