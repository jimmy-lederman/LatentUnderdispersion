# Per-sweep cost of the Section 7.2 (COVID) sampler.
#
#   julia -t 1    --project=. revision2_experiments/covid/profile_baseline.jl
#   julia -t auto --project=. revision2_experiments/covid/profile_baseline.jl
#
# Prints the full-sweep wall clock and allocations, then a sampling-profile breakdown of
# where the time goes. Timings are taken after a warm-up sweep so no compile time is
# included, and the profile runs on the same warmed state.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, Profile

println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  BLAS threads = ", BLAS.get_num_threads())

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
cumdf = cumdf[2:end, 4:end]
data = Dict("Y_NM" => cumdf)
N, M = size(cumdf)
info = Dict("Y0_N" => Int.(reshape(cumdf[:, 1], :, 1)))
println("N = $N, M = $M, cells = $(N*M)")

K, Q, Dmax = 5, 25, 9
model = covid4(N, M, K, Q, Dmax,
               1.0, 1.0, 100.0, 0.1, 0.5, 1.0,      # a b c d g h
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)  # startV1 startV2 alpha beta tauc taud start_tau

function run_sweeps(model, data, info; nwarm=2, ntimed=3, seed=101)
    Random.seed!(seed)
    state = sample_prior(model, info)
    for _ in 1:nwarm
        (_d, state) = backward_sample(model, data, state)
    end
    ts = Float64[]; as = Float64[]
    for i in 1:ntimed
        st = @timed ((_d, state) = backward_sample(model, data, state))
        push!(ts, st.time); push!(as, st.bytes / 2^30)
        @printf("  sweep %d: %6.2f s (%5.2f s gc, %4.1f%%)   %6.2f GiB alloc\n",
                i, st.time, st.gctime, 100*st.gctime/st.time, st.bytes / 2^30)
    end
    return median(ts), median(as), state
end

println("\n--- full backward_sample (2 warm-up sweeps discarded) ---")
titer, aiter, state = run_sweeps(model, data, info)
@printf("median: %.2f s/iter, %.2f GiB/iter\n", titer, aiter)
@printf("4000 sweeps (nburnin=2000, 100x20): %.2f h/chain\n", titer * 4000 / 3600)
@printf("10000 sweeps (nburnin=5000, 5000 kept): %.2f h/chain\n", titer * 10000 / 3600)

println("\n--- sampling profile of one sweep ---")
function profile_one(model, data, state)
    Profile.clear()
    Profile.init(n = 10_000_000, delay = 0.001)
    @profile backward_sample(model, data, state)
    Profile.print(format = :flat, sortedby = :count, mincount = 40, maxdepth = 40)
end
profile_one(model, data, state)
