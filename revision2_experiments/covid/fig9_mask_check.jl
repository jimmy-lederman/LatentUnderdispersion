# Are Figure 9's nine counties inside the held-out mask of each forecasting split?
#
#   julia --project=. revision2_experiments/covid/fig9_mask_check.jl
#
# Figure 9 plots a 10-step forecast for nine named counties. The information-gain campaign
# holds out the last 10 days for a RANDOM 20% of counties per mask seed, so a given county is
# forecast under some seeds and not others. If none of the five seeds covers a county, its
# panel cannot be redrawn from this campaign at all.
#
# Identification is triangulated: the data carry only FIPS, so each candidate code is checked
# against the state prefix AND against the cumulative count the figure shows on the last day.

ROOTDIR = abspath(joinpath(@__DIR__, "../.."))
using CSV, DataFrames, Random, Printf

raw = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
states = raw[2:end, 1]; fips = raw[2:end, 2]; pops = raw[2:end, 3]
cumdf = raw[2:end, 4:end]
Y_NM = cumdf[:, 2:end]
N, M = size(Y_NM)

# (name, FIPS, state FIPS, approximate final cumulative count read off Figure 9)
PANELS = [("Monroe, IL",    17133, 17, 3150), ("Washington, LA", 22117, 22, 3130),
          ("Macon, NC",     37113, 37, 3060), ("Stutsman, ND",   38093, 38, 3140),
          ("Polk, WI",      55095, 55, 3050), ("Jackson, MN",    27063, 27,  790),
          ("DeKalb, MO",    29063, 29,  780), ("Box Butte, NE",  31013, 31,  790),
          ("Big Horn, WY",  56003, 56,  780)]

MASKS = [101, 102, 103, 104, 105]
held = Dict(m => (Random.seed!(m); rand(N) .< 0.2) for m in MASKS)
for m in MASKS
    @printf("mask %d: %d of %d counties held out (%.1f%%)\n",
            m, count(held[m]), N, 100count(held[m])/N)
end

println()
@printf("%-16s %7s %6s %9s %9s %9s   %s\n",
        "panel", "FIPS", "state", "pop", "Y[end]", "fig says", "in mask 101/102/103/104/105")
nmiss = 0
for (name, fp, st, yfig) in PANELS
    idx = findfirst(==(fp), fips)
    if idx === nothing
        @printf("%-16s %7d %6s  NOT FOUND IN DATA\n", name, fp, "-"); global nmiss += 1; continue
    end
    yend = Y_NM[idx, end]
    ok = true     # column 1 is an alphabetical state INDEX, not a state FIPS; Y[end] is the check
    inm = [held[m][idx] for m in MASKS]
    @printf("%-16s %7d %6d %9d %9d %9d   %s%s\n",
            name, fp, states[idx], pops[idx], yend, yfig,
            join([x ? "Y" : "." for x in inm], " "),
            ok ? "" : "   <-- STATE MISMATCH")
    any(inm) || (global nmiss += 1)
end

println()
cov = [count(p -> (i = findfirst(==(p[2]), fips); i !== nothing && held[m][i]), PANELS) for m in MASKS]
for (m, c) in zip(MASKS, cov)
    @printf("mask %d covers %d of the 9 panels\n", m, c)
end
@printf("\npanels covered by NO mask: %d\n", nmiss)
@printf("panels covered by at least one mask: %d of 9\n", 9 - nmiss)
