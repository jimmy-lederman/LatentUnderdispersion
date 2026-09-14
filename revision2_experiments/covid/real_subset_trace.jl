# The new initialisation on a subset of the REAL data: does the growth term plateau, and
# after how many sweeps?
#
#   julia -t 6 --project=. revision2_experiments/covid/real_subset_trace.jl [nsweeps] [nchains] [ncounties]
#
# There is no truth on real data, so no coverage. What we CAN see is the trajectory, which
# is what the open question is about: the published Section 7.2 runs seeded phi at 1 with
# ~5000 sweeps, and on synthetic data that regime was still climbing at 3000. If the same
# is true here, the reported growth term is biased low.
#
# Two arms:
#   seed1     phi seeded at 1  -- what every previous covid run did
#   scalecal  phi's prior draw scale-corrected + the interleaved scale move -- what the run
#             scripts now do
#
# The trace of the mean growth term is printed as the chains run, so a plateau (or its
# absence) is visible rather than inferred from endpoints.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra
using MCMCDiagnosticTools

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
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
K, Q, Dmax = 5, 25, 9
println("N = $N counties, M = $M days, K = $K, Q = $Q, Dmax = $Dmax")
println("$nchains chains x $nsweeps sweeps per arm\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
@printf("mean observed increment = %.2f   mean Y = %.0f   ratio = %.5f\n\n",
        mean(inc), mean(Y_NM), mean(inc)/mean(Y_NM))

Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]
const CHECK = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000]

function run_chain(model, seed, arm)
    Random.seed!(seed)
    st = sample_prior(model, info)
    if arm == :seed1
        st["V_KM"] = fill(1.0, K, M)
    else
        target = max(mean(inc) - st["eps"], 1e-6)
        cur = mean(st["U_NK"] * st["V_KM"])
        st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    end
    trace = Tuple{Int,Float64,Float64,Float64}[]
    mus = Vector{Vector{Float64}}()
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st)
        if s in CHECK || s == nsweeps
            g = mean(st["U_NK"] * st["V_KM"])
            push!(trace, (s, g, st["eps"], mean(st["D_NM"])))
        end
        if s > nsweeps ÷ 2
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            push!(mus, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
        end
    end
    return trace, mus
end

function arm(label, a, smove)
    model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                   0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, smove)
    println("="^92); println(label); println("="^92)
    res = Vector{Any}(undef, nchains)
    for c in 1:nchains
        t = @elapsed res[c] = run_chain(model, 600+c, a)
        @printf("  chain %d done (%.1f min)\n", c, t/60); flush(stdout)
    end
    @printf("\n  %-8s", "sweep")
    for c in 1:nchains; @printf("  chain%d growth", c); end
    @printf("    eps(c1)   meanD(c1)\n")
    tr1 = res[1][1]
    for (i, (s, _, _, _)) in enumerate(tr1)
        @printf("  %-8d", s)
        for c in 1:nchains; @printf("  %13.2f", res[c][1][i][2]); end
        @printf("  %9.3f  %10.2f\n", tr1[i][3], tr1[i][4])
    end
    nk = length(res[1][2])
    A = Array{Float64}(undef, nk, nchains, length(cells))
    for c in 1:nchains, s in 1:nk, p in eachindex(cells); A[s,c,p] = res[c][2][s][p]; end
    r = try maximum(filter(isfinite, [rhat(reshape(A[:,:,p], nk, nchains, 1))[1] for p in eachindex(cells)])) catch; NaN end
    # growth over the last two checkpoints: still climbing, or flat?
    lastg = [res[c][1][end][2] for c in 1:nchains]
    prevg = [res[c][1][end-1][2] for c in 1:nchains]
    @printf("\n  max Rhat on mu over %d cells: %.2f\n", length(cells), r)
    @printf("  growth change over the final checkpoint interval: %s\n",
            join([@sprintf("%+.1f%%", 100*(lastg[c]-prevg[c])/max(prevg[c],1e-9)) for c in 1:nchains], "  "))
    println(); flush(stdout)
end

arm("ARM 1: phi seeded at 1 (what every previous covid run did)", :seed1, false)
arm("ARM 2: scale-corrected prior draw + interleaved scale move (shipped)", :scalecal, true)

println("A growth term still rising at the last checkpoint means the run was too short.")
println("If arm 1 plateaus lower than arm 2, the published runs understated the growth term.")
