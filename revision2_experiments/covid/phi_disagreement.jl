# How different are the chains' factor trajectories, in units a reader of the figure would
# notice? For each factor: the per-chain mean trajectory's peak day, peak height, centre of
# mass and total mass, plus the mean absolute difference between chain-mean trajectories as
# a fraction of the factor's average height. Rhat says "not converged"; this says by how much.
#   julia --project=. phi_disagreement.jl <dir> <lo> <hi> [chains...]
using JLD, Statistics, Printf, LinearAlgebra, Combinatorics
dir = ARGS[1]; lo = parse(Int, ARGS[2]); hi = parse(Int, ARGS[3])
chains = length(ARGS) >= 4 ? parse.(Int, ARGS[4:end]) : [1,2,3,4]
ch = [load(joinpath(dir, "diag_chain$c.jld")) for c in chains]
K = ch[1]["K"]; M = ch[1]["M"]; perms = collect(permutations(1:K))
idx = [findall(s -> lo < s <= hi, c["sw"]) for c in ch]; nk = minimum(length.(idx))
idx = [i[end-nk+1:end] for i in idx]
ph = [[ch[c]["ph"][i] for i in idx[c]] for c in eachindex(ch)]
ref = mean(ph[1])
for _ in 1:8
    global ref; acc = zeros(size(ref)); n = 0
    for c in eachindex(ch), s in 1:nk
        p = perms[argmin([sum(abs2, ref .- ph[c][s][q, :]) for q in perms])]; acc .+= ph[c][s][p, :]; n += 1
    end
    ref = acc ./ n
end
asg = [[perms[argmin([sum(abs2, ref .- ph[c][s][q, :]) for q in perms])] for s in 1:nk] for c in eachindex(ch)]
mean_traj = [[vec(mean(reduce(hcat, [ph[c][s][asg[c][s][k], :] for s in 1:nk]), dims=2)) for c in eachindex(ch)] for k in 1:K]
share = [mean(sum(mean_traj[k][c]) for c in eachindex(ch)) for k in 1:K]
tot = sum(share); ord = sortperm(share, rev=true)
@printf("%s, window %d-%d, chains %s\n\n", dir, lo, hi, string(chains))
for (r, k) in enumerate(ord)
    pk = [argmax(mean_traj[k][c]) for c in eachindex(ch)]
    ph_ = [maximum(mean_traj[k][c]) for c in eachindex(ch)]
    com = [sum(t * mean_traj[k][c][t] for t in 1:M) / sum(mean_traj[k][c]) for c in eachindex(ch)]
    mass = [sum(mean_traj[k][c]) for c in eachindex(ch)]
    # mean |difference| between chain-mean trajectories, relative to the factor's mean height
    d = mean(mean(abs.(mean_traj[k][a] .- mean_traj[k][b])) for a in eachindex(ch) for b in eachindex(ch) if a < b)
    h = mean(mean(mean_traj[k][c]) for c in eachindex(ch))
    @printf("factor %d (share %.2f):  peak day %s  peak height %s\n", r, share[k]/tot,
            join([@sprintf("%d", p) for p in pk], "/"), join([@sprintf("%.0f", x) for x in ph_], "/"))
    @printf("            centre %s   mass %s   mean |chain-mean difference| = %.0f%% of the factor's mean height\n\n",
            join([@sprintf("%.0f", x) for x in com], "/"), join([@sprintf("%.2g", x) for x in mass], "/"), 100d/h)
end
