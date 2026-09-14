# If the factors disagree across chains: is it MULTIMODALITY or SLOW MIXING?
#
#   julia -t 4 --project=. revision2_experiments/covid/mode_structure.jl [nsweeps] [nchains] [ncounties] [K]
#
# Rhat cannot tell these apart, and the fixes are completely different:
#
#   MULTIMODALITY  the posterior has genuinely distinct ways to split the data into K waves
#                  and each chain settles into one. Each chain is STATIONARY; the chains
#                  differ. Fix: reduce K (chosen by held-out density), or add structure that
#                  breaks the symmetry between factors. Running longer does nothing.
#
#   SLOW MIXING    one basin, chains still travelling. Each chain is still DRIFTING.
#                  Fix: sampler work, or many more sweeps. Reducing K does nothing.
#
# The discriminator is drift-within-chain versus spread-between-chains, on the fitted
# surface mu = theta'phi (which is already invariant to the per-factor scale and to label
# permutation, so any disagreement in it is real):
#
#   drift small, between-chain spread large  -> MULTIMODAL (each chain settled elsewhere)
#   drift large                              -> SLOW MIXING (nobody has settled yet)
#
# The pairwise-distance histogram adds a second read: distinct modes show up as clustered
# distances with a gap, slow mixing as a continuous spread.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2500
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 150
K       = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
Q, Dmax = 25, 9
println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  K = ", K)

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
println("N = $N, M = $M, $nchains chains x $nsweeps sweeps\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true)

Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:400]

# Return the per-cell mu averaged over the 3rd quarter and over the 4th quarter separately,
# so within-chain drift can be compared against between-chain spread on the same scale.
function run_chain(seed)
    Random.seed!(seed)
    st = sample_prior(model, info)
    target = max(mean_inc - st["eps"], 1e-6)
    cur = mean(st["U_NK"] * st["V_KM"])
    st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    q3 = zeros(length(cells)); q4 = zeros(length(cells)); n3 = 0; n4 = 0
    sq = zeros(length(cells))
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st)
        if s > nsweeps ÷ 2
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            v = Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells]
            sq .+= v .^ 2
            if s <= 3*nsweeps ÷ 4
                q3 .+= v; n3 += 1
            else
                q4 .+= v; n4 += 1
            end
        end
    end
    q3 ./= n3; q4 ./= n4
    mn = (q3 .* n3 .+ q4 .* n4) ./ (n3 + n4)
    sd = sqrt.(max.(sq ./ (n3 + n4) .- mn .^ 2, 0.0))
    return mn, q3, q4, sd
end

res = Vector{Any}(undef, nchains)
for c in 1:nchains
    t = @elapsed res[c] = run_chain(1200 + c)
    @printf("  chain %2d done (%.1f min)\n", c, t/60); flush(stdout)
end

MN = reduce(hcat, [r[1] for r in res])          # cells x chains
Q3 = reduce(hcat, [r[2] for r in res])
Q4 = reduce(hcat, [r[3] for r in res])
SD = reduce(hcat, [r[4] for r in res])

# within-chain drift, in units of that chain's own posterior sd for the cell
drift = mean(abs.(Q4 .- Q3) ./ max.(SD, 1e-9))
# between-chain spread, same units
between = mean(std(MN, dims=2) ./ max.(mean(SD, dims=2), 1e-9))

println()
@printf("within-chain drift  (|Q4-Q3| / sd) : %6.3f\n", drift)
@printf("between-chain spread (sd_chains/sd): %6.3f\n", between)
@printf("ratio between/within               : %6.2f\n", between / max(drift, 1e-9))

# pairwise distances between chain-mean surfaces, scaled
d = Float64[]
for a in 1:nchains, b in (a+1):nchains
    push!(d, sqrt(mean((MN[:,a] .- MN[:,b]).^2)) / mean(mean(SD, dims=2)))
end
sort!(d)
@printf("\npairwise chain distances (in posterior sd units), %d pairs:\n", length(d))
@printf("  min %.2f  q25 %.2f  median %.2f  q75 %.2f  max %.2f\n",
        d[1], quantile(d, .25), median(d), quantile(d, .75), d[end])
gap = maximum(diff(d)) / max(median(d), 1e-9)
@printf("  largest relative gap between consecutive distances: %.2f\n", gap)

println("""
READING
  drift >> between            -> chains still travelling: SLOW MIXING, sampler work or
                                 many more sweeps; reducing K will not help.
  drift small, between large  -> each chain settled somewhere different: MULTIMODAL;
                                 reduce K (by held-out density) or break the symmetry.
                                 Running longer will not help.
  a large gap in the pairwise distances is further evidence of distinct modes rather
  than a continuous spread.""")
