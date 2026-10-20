# Per-FACTOR diagnostics: for each factor k, Rhat and ESS over the M elements of its scaled
# trajectory s_k*phi_k and over its N (or n_tracked) normalised loadings theta_{.,k}, after
# aligning labels with one permutation per chain (the per-draw assignment is constant within
# a chain; see analyze_diag.jl). Factors are listed in descending share, with their centre
# of mass and cross-chain cosine similarity, so the ones the paper shows can be read off.
#
#   julia --project=. per_factor.jl <dir> [window_lo] [window_hi] [chains...]
using JLD, Statistics, Printf, LinearAlgebra, Combinatorics, MCMCDiagnosticTools
dir = ARGS[1]
chains = length(ARGS) >= 4 ? parse.(Int, ARGS[4:end]) : [1, 2, 3, 4]
ch = [load(joinpath(dir, "diag_chain$c.jld")) for c in chains]
K = ch[1]["K"]; M = ch[1]["M"]; sw = ch[1]["sw"]
lo = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (maximum(c["sw"][end] for c in ch) ÷ 3)
hi = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : minimum(c["sw"][end] for c in ch)
idx = [findall(s -> lo < s <= hi, c["sw"]) for c in ch]
nk = minimum(length.(idx)); idx = [i[end-nk+1:end] for i in idx]
@printf("%s\nchains %s, window %d-%d, %d draws each\n", dir, string(chains), lo, hi, nk)
perms = collect(permutations(1:K))
ph = [[ch[c]["ph"][i] for i in idx[c]] for c in eachindex(ch)]
th = [[ch[c]["th"][i] for i in idx[c]] for c in eachindex(ch)]
ref = mean(ph[1])
for _ in 1:8                                  # align by the trajectories, per draw
    global ref
    acc = zeros(size(ref)); n = 0
    for c in eachindex(ch), s in 1:nk
        p = perms[argmin([sum(abs2, ref .- ph[c][s][q, :]) for q in perms])]
        acc .+= ph[c][s][p, :]; n += 1
    end
    ref = acc ./ n
end
asg = [[perms[argmin([sum(abs2, ref .- ph[c][s][q, :]) for q in perms])] for s in 1:nk] for c in eachindex(ch)]
cp = [asg[c][argmax([count(==(p), asg[c]) for p in asg[c]])] for c in eachindex(ch)]
@printf("per-chain modal permutation held in %s%% of draws\n\n",
        join([@sprintf("%.0f", 100 * count(==(cp[c]), asg[c]) / nk) for c in eachindex(ch)], "/"))
r3(x) = reshape(x, size(x, 1), size(x, 2), 1)
nI = size(th[1][1], 1)
PH = [Array{Float64}(undef, nk, length(ch), M) for _ in 1:K]
TH = [Array{Float64}(undef, nk, length(ch), nI) for _ in 1:K]
for c in eachindex(ch), s in 1:nk, k in 1:K
    p = asg[c][s]
    PH[k][s, c, :] = ph[c][s][p[k], :]
    TH[k][s, c, :] = th[c][s][:, p[k]]
end
share = [mean(sum(PH[k][s, c, :]) / sum(sum(PH[j][s, c, :]) for j in 1:K) for s in 1:nk, c in eachindex(ch)) for k in 1:K]
com   = [mean(sum(t * PH[k][s, c, t] for t in 1:M) / sum(PH[k][s, c, :]) for s in 1:nk, c in eachindex(ch)) for k in 1:K]
ord = sortperm(share, rev=true)
cosr(a, b) = dot(a, b) / (norm(a) * norm(b))
@printf("%-7s %6s %7s | %-22s | %-22s | %s\n", "factor", "share", "centre",
        "s_k*phi_k: Rhat med/max (ESS/chain)", "theta_.k: Rhat med/max (ESS/chain)", "min pairwise cosine of chain means")
for (rank, k) in enumerate(ord)
    rp = [rhat(r3(PH[k][:, :, t]))[1] for t in 1:M]; ep = [ess(r3(PH[k][:, :, t]); kind=:bulk)[1] for t in 1:M]
    rt = [rhat(r3(TH[k][:, :, i]))[1] for i in 1:nI]; et = [ess(r3(TH[k][:, :, i]); kind=:bulk)[1] for i in 1:nI]
    mu_c = [vec(mean(PH[k][:, c, :], dims=1)) for c in eachindex(ch)]
    mincos = minimum(cosr(mu_c[a], mu_c[b]) for a in eachindex(ch) for b in eachindex(ch) if a < b)
    @printf("%-7d %6.2f %7.0f | %6.2f/%6.2f (%6.0f)     | %6.2f/%6.2f (%6.0f)     | %.3f\n",
            rank, share[k], com[k], median(rp), maximum(rp), median(ep)/length(ch),
            median(rt), maximum(rt), median(et)/length(ch), mincos)
end
