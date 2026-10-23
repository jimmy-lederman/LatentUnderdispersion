# Reported diagnostics for the finished information-gain cells, with factor labels ALIGNED.
#
#   julia --project=. revision2_experiments/covid/inspect_diag.jl <infogain_dir> [arm]
#
# Labels do not switch within a Gibbs chain but are arbitrary ACROSS chains, so any
# cross-chain comparison of theta or phi has to align them first -- one permutation per
# chain, chosen on the phi trajectories (FINDINGS: aligning on theta is ambiguous between
# similar factors and inflates Rhat). Without this step the factors look unrelated.

using JLD, Statistics, Printf, LinearAlgebra, Combinatorics, MCMCDiagnosticTools

DIR = length(ARGS) >= 1 ? ARGS[1] : "output/covid/infogain"
ARM = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
MASKS = [101, 102, 103, 104, 105]
cf(d,m) = (p = joinpath(DIR,"D$(d)_mask$(m)"); isdir(p) ?
    sort(filter(f -> occursin(r"summary_chain\d+\.jld$", f), readdir(p; join=true))) : String[])

function chain(f)
    r = load(f); keep = findall(>(r["burnin"]), r["sw"])
    (ll = Float64.(r["ll"][keep]), dm = Float64.(r["dm"][keep]),
     mu = r["mu"][keep], dc = r["dc"][keep], ph = r["ph"][keep], n = length(keep))
end
r3(A) = reshape(A, size(A,1), size(A,2), 1)
safe(f,x) = try f(x) catch; NaN end
rh(M) = safe(_ -> rhat(r3(M))[1], 0)
esf(v) = safe(x -> ess(r3(reshape(x,:,1)); kind=:bulk)[1], v)
padm(cols) = (n = minimum(length.(cols)); reduce(hcat, [Float64.(c[1:n]) for c in cols]))

# mean phi per chain -> permutation of each chain onto chain 1
function alignperms(ch)
    K = size(ch[1].ph[1], 1)
    mp = [reduce(+, Float64.(c.ph[i]) for i in eachindex(c.ph)) ./ length(c.ph) for c in ch]
    ref = mp[1]
    perms = [collect(1:K)]
    for c in 2:length(ch)
        best, bestscore = collect(1:K), -Inf
        for p in permutations(1:K)
            sc = sum(dot(ref[k,:], mp[c][p[k],:]) / (norm(ref[k,:])*norm(mp[c][p[k],:]) + 1e-300) for k in 1:K)
            sc > bestscore && (bestscore = sc; best = collect(p))
        end
        push!(perms, best)
    end
    (perms, mp)
end

for m in MASKS
    fs = cf(ARM, m); length(fs) >= 2 || continue
    ch = [chain(f) for f in fs]; C = length(ch); K = size(ch[1].ph[1],1)
    nmin = minimum(c.n for c in ch)
    println("="^78)
    @printf("D = %d, mask %d -- %d chains x %d post-burn-in draws\n", ARM, m, C, nmin)

    L = padm([c.ll for c in ch]); Dm = padm([c.dm for c in ch])
    P = length(ch[1].mu[1])
    MU = Array{Float64}(undef, nmin, C, P); DD = Array{Float64}(undef, nmin, C, P)
    for c in 1:C, s in 1:nmin, p in 1:P
        MU[s,c,p] = ch[c].mu[s][p]; DD[s,c,p] = Float64(ch[c].dc[s][p])
    end
    mur = filter(isfinite, [safe(_ -> rhat(r3(MU[:,:,p]))[1], p) for p in 1:P])
    mue = [esf(MU[:,c,p]) for c in 1:C, p in 1:P]
    dvar = [p for p in 1:P if std(vec(DD[:,:,p])) > 0]
    dr = filter(isfinite, [safe(_ -> rhat(r3(DD[:,:,p]))[1], p) for p in dvar])

    perms, mp = alignperms(ch)
    PH = Array{Float64}(undef, nmin, C, K*size(ch[1].ph[1],2))
    for c in 1:C, s in 1:nmin
        A = Float64.(ch[c].ph[s])[perms[c], :]
        PH[s,c,:] = vec(A)
    end
    phr = filter(isfinite, [safe(_ -> rhat(r3(PH[:,:,p]))[1], p) for p in 1:size(PH,3)])
    phe = [esf(PH[:,c,p]) for c in 1:C, p in 1:size(PH,3)]

    @printf("  %-26s %10s %12s %12s\n", "quantity", "Rhat med", "Rhat max", "ESS/chain")
    @printf("  %-26s %10.2f %12s %12.0f\n", "held-out log density", rh(L), "-",
            mean(filter(isfinite, [esf(L[:,c]) for c in 1:C])))
    @printf("  %-26s %10.2f %12s %12s\n", "mean D", rh(Dm), "-", "-")
    @printf("  %-26s %10.2f %12.2f %12.0f\n", "mu_it (300 cells)", median(mur), maximum(mur),
            mean(filter(isfinite, mue)))
    isempty(dr) || @printf("  %-26s %10.2f %12.2f %12s   (%d of %d cells vary)\n",
            "D_it (varying cells)", median(dr), maximum(dr), "-", length(dvar), P)
    @printf("  %-26s %10.2f %12.2f %12.0f\n", "s_k*phi_kt (ALIGNED)", median(phr), maximum(phr),
            mean(filter(isfinite, phe)))

    println("  per-factor, after alignment:")
    for k in 1:K
        traj = [mp[c][perms[c][k], :] for c in 1:C]
        cosmin = minimum(dot(traj[a],traj[b])/(norm(traj[a])*norm(traj[b])+1e-300) for a in 1:C, b in 1:C)
        @printf("    k=%d  peak day %-18s share %-20s min cosine %.3f\n", k,
                join([argmax(t) for t in traj], "/"),
                join([@sprintf("%.2f", sum(t)/sum(sum.(traj[ci] for ci in 1:C))*C) for t in traj], "/"),
                cosmin)
    end
    println("  permutation applied per chain: ", join([join(p, "") for p in perms], "  "))
end
