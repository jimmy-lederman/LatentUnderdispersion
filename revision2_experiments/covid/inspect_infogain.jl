# What do the finished information-gain cells actually say?
#
#   julia --project=. revision2_experiments/covid/inspect_infogain.jl <infogain_dir>
#
# Reports, for whatever is complete: the diagnostics Section 7.2 reports, the information gain
# for any mask where both the inferred arm and the D = 1 baseline are done, and the raw
# material for Figures 9 (forecast intervals) and 10 (factor structure).
#
# NOTE the key names are the RAW ones run_diag_chain.jl writes -- ll / mu / dm / ph / th --
# not the aliases summarize_covid.jl maps them to. Every per-draw array covers EVERY sweep,
# so the burn-in window has to be applied here.

using JLD, Statistics, Printf, LinearAlgebra, LogExpFunctions, MCMCDiagnosticTools

DIR = length(ARGS) >= 1 ? ARGS[1] : "output/covid/infogain"
MASKS = [101, 102, 103, 104, 105]; ARMS = [0, 1, 3, 5, 7, 9]
celldir(d, m) = joinpath(DIR, "D$(d)_mask$(m)")
cellfiles(d, m) = isdir(celldir(d,m)) ?
    sort(filter(f -> occursin(r"summary_chain\d+\.jld$", f), readdir(celldir(d,m); join=true))) : String[]
ncell(d, m) = length(cellfiles(d, m))

# load one chain, dropping warm-up from the per-draw arrays
function chain(f)
    r = load(f)
    keep = findall(>(r["burnin"]), r["sw"])
    (ll = Float64.(r["ll"][keep]), dm = Float64.(r["dm"][keep]),
     mu = r["mu"][keep], ph = r["ph"][keep],
     pc = r["heldout_logmeanexp_cells"], S1th = r["S1th"], segn = r["segn"],
     fc = get(r, "fc", Vector{Matrix{Int32}}()), fcc = get(r, "fc_counties", Int[]),
     nkeep = length(keep))
end
r3(A) = reshape(A, size(A,1), size(A,2), 1)
safe(f, x) = try f(x) catch; NaN end
rh(M) = safe(_ -> rhat(r3(M))[1], 0)
essof(v) = safe(x -> ess(r3(reshape(x,:,1)); kind=:bulk)[1], v)
function pad(cols)
    n = minimum(length.(cols)); reduce(hcat, [Float64.(c[1:n]) for c in cols])
end

println("="^80); println("COMPLETE CELLS  (4 = all chains)")
for d in ARMS
    @printf("  D=%-2s %s\n", d, join([lpad(string(ncell(d,m), "/4"), 7) for m in MASKS]))
end

println("\n", "="^80)
println("DIAGNOSTICS -- inferred-D arm (D = 0), per cell with >= 2 chains")
@printf("%-8s %3s %9s %8s %11s %9s %9s %8s\n",
        "mask","ch","draws/ch","ll Rhat","ll ESS/ch","mu Rhat","mu ESS/ch","meanD Rh")
for m in MASKS
    fs = cellfiles(0, m); length(fs) >= 2 || continue
    ch = [chain(f) for f in fs]
    L  = pad([c.ll for c in ch]); Dm = pad([c.dm for c in ch])
    nmin = minimum(c.nkeep for c in ch); P = length(ch[1].mu[1])
    MU = Array{Float64}(undef, nmin, length(ch), P)
    for c in eachindex(ch), s in 1:nmin, p in 1:P; MU[s,c,p] = ch[c].mu[s][p]; end
    mur = filter(isfinite, [safe(_ -> rhat(r3(MU[:,:,p]))[1], p) for p in 1:P])
    mue = mean(filter(isfinite, [essof(MU[:,c,p]) for c in eachindex(ch), p in 1:P]))
    @printf("%-8s %3d %9d %8.2f %11.0f %9.2f %9.0f %8.2f\n",
            "m$m", length(ch), nmin, rh(L), mean(filter(isfinite, [essof(L[:,c]) for c in 1:size(L,2)])),
            median(mur), mue, rh(Dm))
end

println("\n", "="^80)
println("INFORMATION GAIN over the D = 1 baseline")
inforate(d,m) = (pc = [chain(f).pc for f in cellfiles(d,m)];
                 H = length(pc[1]);
                 mean(logsumexp([pc[c][h] for c in eachindex(pc)]) - log(length(pc)) for h in 1:H))
gains = Float64[]
@printf("%-8s %6s %14s %14s %12s\n", "mask","chains","D inferred","D = 1","gain")
for m in MASKS
    (ncell(0,m) >= 2 && ncell(1,m) >= 2) || continue
    a = inforate(0,m); b = inforate(1,m); push!(gains, a-b)
    @printf("%-8s %6s %14.4f %14.4f %12.4f\n", "m$m", "$(ncell(0,m))/$(ncell(1,m))", a, b, a-b)
end
if !isempty(gains)
    @printf("  mean gain over %d split(s): %.4f", length(gains), mean(gains))
    length(gains) > 1 && @printf("  (1.96 SE %.4f)", 1.96*std(gains)/sqrt(length(gains)))
    println("\n  paper's current table: 0.367 +/- 0.006 over 5 splits")
end
for m in MASKS
    (ncell(3,m) >= 2 && ncell(1,m) >= 2) && @printf("  D=3 vs D=1, m%d: %.4f\n", m, inforate(3,m)-inforate(1,m))
end

println("\n", "="^80)
println("FIGURE 10 -- factor structure, inferred arm (scaled trajectory s_k*phi_kt)")
for m in MASKS
    fs = cellfiles(0, m); length(fs) >= 2 || continue
    ch = [chain(f) for f in fs]; K = size(ch[1].ph[1], 1)
    # per-chain mean trajectory, then peak day / share / cross-chain agreement
    means = [reduce(+, Float64.(c.ph[i]) for i in eachindex(c.ph)) ./ length(c.ph) for c in ch]
    tot   = [sum(M) for M in means]
    @printf("  mask %d (%d chains):\n", m, length(ch))
    for k in 1:K
        peaks  = [argmax(M[k,:]) for M in means]
        shares = [sum(M[k,:])/tot[ci] for (ci,M) in enumerate(means)]
        cosmin = minimum(dot(means[a][k,:], means[b][k,:]) /
                         (norm(means[a][k,:])*norm(means[b][k,:]))
                         for a in 1:length(means), b in 1:length(means))
        @printf("    k=%d  peak day %-22s share %-22s min cosine %.3f\n", k,
                join(peaks, "/"), join([@sprintf("%.2f", s) for s in shares], "/"), cosmin)
    end
end

println("\n", "="^80)
println("FIGURE 9 -- 10-step forecast intervals vs truth, the nine panels (mask 101)")
PANELS = [("Monroe, IL",17133),("Washington, LA",22117),("Macon, NC",37113),
          ("Stutsman, ND",38093),("Polk, WI",55095),("Jackson, MN",27063),
          ("DeKalb, MO",29063),("Box Butte, NE",31013),("Big Horn, WY",56003)]
if ncell(0,101) >= 1
    ch = [chain(f) for f in cellfiles(0,101)]
    ch1 = [chain(f) for f in cellfiles(1,101)]
    fcc = ch[1].fcc
    if isempty(ch[1].fc)
        println("  (no forecast draws stored in these files)")
    else
        @printf("  %d chains inferred / %d chains D=1; %d forecast records/chain, horizon %d\n",
                length(ch), length(ch1), length(ch[1].fc), size(ch[1].fc[1],2))
        @printf("  %-16s %10s %24s %24s\n", "panel","truth d10","inferred 95% CI (last day)","D=1 95% CI (last day)")
        # county row index -> position in the fc matrix
        using CSV, DataFrames
        raw = Matrix(CSV.read(joinpath(dirname(DIR),"..","..","data","covid","usafull_final.csv"), DataFrame))
        fips = raw[2:end,2]; Yd = raw[2:end,5:end]
        for (nm, fp) in PANELS
            gi = findfirst(==(fp), fips); gi === nothing && continue
            pos = findfirst(==(gi), fcc); pos === nothing && continue
            draws_a = vcat([Float64.(M[pos,end]) for c in ch for M in c.fc]...)
            draws_b = isempty(ch1) ? Float64[] : vcat([Float64.(M[pos,end]) for c in ch1 for M in c.fc]...)
            qa = quantile(draws_a, [0.025,0.975])
            qb = isempty(draws_b) ? (NaN,NaN) : quantile(draws_b, [0.025,0.975])
            @printf("  %-16s %10d %11.0f - %-10.0f %11.0f - %-10.0f\n",
                    nm, Yd[gi,end], qa[1], qa[2], qb[1], qb[2])
        end
    end
else
    println("  mask 101 inferred arm has no finished chains yet")
end
