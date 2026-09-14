# Is Dmax binding? Does the inferred order settle, or press against its own ceiling?
#
#   julia -t 6 --project=. revision2_experiments/covid/dmax_ladder.jl [nsweeps] [nchains] [ncounties]
#
# On the real subset, mean D climbs monotonically 5.32 -> 7.60 over 5000 sweeps with no sign
# of flattening, in BOTH initialisation arms, against Dmax = 9. Two very different readings:
#
#   NOT CONVERGED   D is a lagging indicator of the per-cell factor fit -- a cell can only
#                   support a large order once its rate is accurate -- and the per-cell mu
#                   is still at Rhat ~2.4 when the aggregate growth has already plateaued.
#                   Then D settles somewhere below Dmax given enough sweeps.
#
#   CENSORED        the data want an order above 9, and Dmax is truncating the answer. That
#                   is the same shape as the MedNB dispersion-floor issue in Section 6.3:
#                   a reported parameter reflecting a fixed bound rather than the data.
#
# Running the same data at two values of Dmax separates them. If mean D lands in the same
# place regardless, the bound is slack and it is a convergence question. If it tracks Dmax
# upward, and the fraction of cells sitting AT Dmax stays high, the bound is binding.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra
using MCMCDiagnosticTools

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 15000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200
println("Julia ", VERSION, "  threads = ", Threads.nthreads())

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
K, Q = 5, 25
println("N = $N counties, M = $M days, K = $K, Q = $Q")
println("$nchains chains x $nsweeps sweeps per Dmax\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)

Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]
const CHECK = [250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000, 20000, 30000]

function run_chain(model, seed, Dmax)
    Random.seed!(seed)
    st = sample_prior(model, info)
    target = max(mean_inc - st["eps"], 1e-6)
    cur = mean(st["U_NK"] * st["V_KM"])
    st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    trace = Tuple{Int,Float64,Float64,Float64,Float64}[]
    mus = Vector{Vector{Float64}}()
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st)
        if s in CHECK || s == nsweeps
            D = st["D_NM"]
            push!(trace, (s, mean(D), count(==(Dmax), D)/length(D),
                          count(==(1), D)/length(D), mean(st["U_NK"] * st["V_KM"])))
        end
        s > nsweeps ÷ 2 && push!(mus,
            Float64[sum(st["U_NK"][n,:] .* st["V_KM"][:,m]) for (n,m) in cells])
    end
    return trace, mus
end

function ladder(Dmax)
    model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                   0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true)
    println("="^96); println("Dmax = $Dmax"); println("="^96)
    res = Vector{Any}(undef, nchains)
    for c in 1:nchains
        t = @elapsed res[c] = run_chain(model, 900+c, Dmax)
        @printf("  chain %d done (%.1f min)\n", c, t/60); flush(stdout)
    end
    @printf("\n  %-7s %10s %12s %12s %12s\n", "sweep", "mean D", "frac at Dmax", "frac at D=1", "growth")
    for (i, tr) in enumerate(res[1][1])
        s = tr[1]
        md = mean(res[c][1][i][2] for c in 1:nchains)
        fm = mean(res[c][1][i][3] for c in 1:nchains)
        f1 = mean(res[c][1][i][4] for c in 1:nchains)
        gg = mean(res[c][1][i][5] for c in 1:nchains)
        @printf("  %-7d %10.3f %11.1f%% %11.1f%% %12.2f\n", s, md, 100fm, 100f1, gg)
    end
    nk = length(res[1][2])
    A = Array{Float64}(undef, nk, nchains, length(cells))
    for c in 1:nchains, s in 1:nk, p in eachindex(cells); A[s,c,p] = res[c][2][s][p]; end
    r = try maximum(filter(isfinite, [rhat(reshape(A[:,:,p], nk, nchains, 1))[1] for p in eachindex(cells)])) catch; NaN end
    lastD = [res[c][1][end][2] for c in 1:nchains]
    prevD = [res[c][1][end-1][2] for c in 1:nchains]
    @printf("\n  max Rhat on mu over %d cells: %.2f\n", length(cells), r)
    @printf("  mean D change over the final checkpoint interval: %s\n",
            join([@sprintf("%+.2f%%", 100*(lastD[c]-prevD[c])/prevD[c]) for c in 1:nchains], "  "))
    println(); flush(stdout)
end

ladder(9)
ladder(15)

println("Same landing point at both Dmax => the bound is slack and this is a convergence")
println("question. mean D tracking Dmax upward, with a large fraction of cells sitting AT")
println("Dmax, => the bound is censoring the reported order.")
