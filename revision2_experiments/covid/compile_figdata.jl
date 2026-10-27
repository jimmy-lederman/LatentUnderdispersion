# Extract the Figure 9 and Figure 10 inputs from the information-gain grid.
#
#   julia --project=. revision2_experiments/covid/compile_figdata.jl <infogain_dir> <outdir>
#
# Writes CSVs, because nothing in the repository currently turns these runs into the two
# figures -- the versions in draft/figures/covid/ came from an ad-hoc session and cannot be
# regenerated. These files are the reproducible input a plotting script can consume.
#
#   fig9_forecasts.csv   county, state, day, arm, truth, lo95, median, hi95
#   fig10_trajectories.csv  factor, day, mean, lo95, hi95        (pooled over aligned chains)
#   fig10_loadings.csv      fips, factor, loading                (posterior mean theta/s_k)
#
# Both figures use the inferred arm; Figure 9 also needs the D = 1 arm for its Poisson
# comparison. Factor labels are ALIGNED across chains on the phi trajectories before pooling
# -- without that the chains' k=1 are unrelated objects and any pooled factor is meaningless.

using JLD, Statistics, Printf, LinearAlgebra, Combinatorics, CSV, DataFrames

DIR  = ARGS[1]
OUT  = ARGS[2]; mkpath(OUT)
ROOT = abspath(joinpath(@__DIR__, "../.."))
MASK_FIG9 = 101          # the only split containing all nine Figure 9 counties

cf(d,m) = (p = joinpath(DIR,"D$(d)_mask$(m)"); isdir(p) ?
    sort(filter(f -> occursin(r"summary_chain\d+\.jld$", f), readdir(p; join=true))) : String[])

function load_chain(f)
    r = load(f)
    keep  = findall(>(r["burnin"]), r["sw"])          # per-sweep arrays
    fkeep = findall(>(r["burnin"]), r["hosw"])        # the HOTHIN-cadence arrays
    (ph = r["ph"][keep],
     fc = isempty(r["fc"]) ? Matrix{Int32}[] : r["fc"][fkeep],
     fcc = get(r, "fc_counties", Int[]),
     S1th = r["S1th"], segn = r["segn"])
end

# one permutation per chain, chosen on the mean phi trajectories, onto chain 1
function align(chs)
    K = size(chs[1].ph[1], 1)
    mp = [reduce(+, Float64.(c.ph[i]) for i in eachindex(c.ph)) ./ length(c.ph) for c in chs]
    perms = [collect(1:K)]
    for c in 2:length(chs)
        best, bs = collect(1:K), -Inf
        for p in permutations(1:K)
            s = sum(dot(mp[1][k,:], mp[c][p[k],:]) /
                    (norm(mp[1][k,:])*norm(mp[c][p[k],:]) + 1e-300) for k in 1:K)
            s > bs && (bs = s; best = collect(p))
        end
        push!(perms, best)
    end
    (perms, mp)
end

# ---------------- Figure 10: trajectories and loadings ------------------------------
println("Figure 10: pooling the inferred arm over masks with 4 chains")
traj_rows = DataFrame(factor=Int[], day=Int[], mean=Float64[], lo95=Float64[], hi95=Float64[])
load_rows = DataFrame(fips=Int[], factor=Int[], loading=Float64[])
raw  = Matrix(CSV.read(joinpath(ROOT,"data/covid/usafull_final.csv"), DataFrame))
fips = Int.(raw[2:end,2]); Ydat = raw[2:end,5:end]

m = MASK_FIG9
fs = cf(0, m)
if length(fs) >= 2
    chs = [load_chain(f) for f in fs]
    perms, _ = align(chs)
    K, M = size(chs[1].ph[1])
    # pooled posterior draws of the aligned trajectory
    for k in 1:K, t in 1:M
        v = Float64[]
        for (ci, c) in enumerate(chs), s in eachindex(c.ph)
            push!(v, Float64(c.ph[s][perms[ci][k], t]))
        end
        q = quantile(v, [0.025, 0.975])
        push!(traj_rows, (k, t, mean(v), q[1], q[2]))
    end
    # posterior-mean loadings for every county, aligned then averaged over chains
    nz = findall(>(0), chs[1].segn)
    acc = zeros(Float64, length(fips), K)
    for (ci, c) in enumerate(chs)
        th = dropdims(sum(Float64, c.S1th[:,:,nz]; dims=3); dims=3) ./ sum(c.segn[nz])
        acc .+= th[:, perms[ci]]
    end
    acc ./= length(chs)
    for i in eachindex(fips), k in 1:K
        push!(load_rows, (fips[i], k, acc[i,k]))
    end
    @printf("  mask %d: %d chains, K=%d, M=%d -> %d trajectory rows, %d loading rows\n",
            m, length(chs), K, M, nrow(traj_rows), nrow(load_rows))
end
CSV.write(joinpath(OUT,"fig10_trajectories.csv"), traj_rows)
CSV.write(joinpath(OUT,"fig10_loadings.csv"), load_rows)

# ---------------- Figure 9: forecast intervals -------------------------------------
println("Figure 9: forecast intervals for the nine panels, mask $MASK_FIG9")
PANELS = [("Monroe","IL",17133),("Washington","LA",22117),("Macon","NC",37113),
          ("Stutsman","ND",38093),("Polk","WI",55095),("Jackson","MN",27063),
          ("DeKalb","MO",29063),("Box Butte","NE",31013),("Big Horn","WY",56003)]
f9 = DataFrame(county=String[], state=String[], day=Int[], arm=String[],
               truth=Int[], lo95=Float64[], median=Float64[], hi95=Float64[])
for (armlabel, d) in (("inferred", 0), ("poisson_D1", 1))
    fs = cf(d, MASK_FIG9); isempty(fs) && continue
    chs = [load_chain(f) for f in fs]
    isempty(chs[1].fc) && (println("  no forecast draws for arm $armlabel"); continue)
    fcc = chs[1].fcc; H = size(chs[1].fc[1], 2); M = size(Ydat, 2)
    for (nm, st, fp) in PANELS
        gi  = findfirst(==(fp), fips);  gi  === nothing && continue
        pos = findfirst(==(gi), fcc);   pos === nothing && continue
        for h in 1:H
            v = Float64[]
            for c in chs, Mx in c.fc; push!(v, Float64(Mx[pos, h])); end
            q = quantile(v, [0.025, 0.5, 0.975])
            push!(f9, (nm, st, M - H + h, armlabel, Int(Ydat[gi, M - H + h]), q[1], q[2], q[3]))
        end
    end
    @printf("  arm %-11s %d chains x %d forecast records\n", armlabel, length(chs), length(chs[1].fc))
end
CSV.write(joinpath(OUT,"fig9_forecasts.csv"), f9)
@printf("\nwrote %d fig9 rows, %d fig10 trajectory rows, %d fig10 loading rows -> %s\n",
        nrow(f9), nrow(traj_rows), nrow(load_rows), OUT)
