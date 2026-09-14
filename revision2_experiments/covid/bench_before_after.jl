# Interleaved before/after timing of the Section 7.2 sweep on the real data.
#
#   julia -t 1 --project=. revision2_experiments/covid/bench_before_after.jl
#   julia -t 8 --project=. revision2_experiments/covid/bench_before_after.jl
#
# Old and new are run ALTERNATELY and medians are reported, because absolute wall-clock on
# this machine drifts upward over a session of sustained benchmarking; running all the old
# replicates first and all the new ones after would credit the change with the drift.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
include(joinpath(@__DIR__, "reference/covid_orig.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra

println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  cpu = ", Sys.CPU_NAME)

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
cumdf = cumdf[2:end, 4:end]
data = Dict("Y_NM" => cumdf)
N, M = size(cumdf)
info = Dict("Y0_N" => Int.(reshape(cumdf[:, 1], :, 1)))
K, Q, Dmax = 5, 25, 9
hp = (1.0, 1.0, 100.0, 0.1, 0.5, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
mnew  = covid4(N, M, K, Q, Dmax, hp...)
morig = covid4orig(N, M, K, Q, Dmax, hp...)
println("N = $N, M = $M, cells = $(N*M), K = $K, Q = $Q, Dmax = $Dmax")

# Two warm-up sweeps per version: one for compilation, one so both are timed from a state
# that has already been through the sampler rather than straight from the prior.
function warmed(model, data, info, seed)
    Random.seed!(seed)
    st = sample_prior(model, info)
    for _ in 1:2; (_d, st) = backward_sample(model, data, st); end
    return st
end
println("warming up...")
snew  = warmed(mnew,  data, info, 101)
sorig = warmed(morig, data, info, 101)

function interleave(morig, mnew, data, sorig, snew, nrep)
    told = Float64[]; tnew = Float64[]; aold = Float64[]; anew = Float64[]
    for i in 1:nrep
        r = @timed ((_d, sorig) = backward_sample(morig, data, sorig))
        push!(told, r.time); push!(aold, r.bytes/2^30)
        r = @timed ((_d, snew) = backward_sample(mnew, data, snew))
        push!(tnew, r.time); push!(anew, r.bytes/2^30)
        @printf("  rep %d:  old %6.2f s / %5.2f GiB    new %6.2f s / %5.2f GiB\n",
                i, told[end], aold[end], tnew[end], anew[end])
    end
    return told, tnew, aold, anew
end
told, tnew, aold, anew = interleave(morig, mnew, data, sorig, snew, 3)

mo, mn = median(told), median(tnew)
@printf("\nmedian per sweep:  old %.2f s   new %.2f s   speedup %.2fx\n", mo, mn, mo/mn)
@printf("allocations:       old %.2f GiB new %.2f GiB reduction %.2fx\n",
        median(aold), median(anew), median(aold)/median(anew))
for s in (4000, 10000)
    @printf("%5d sweeps/chain:  old %5.2f h   new %5.2f h\n", s, mo*s/3600, mn*s/3600)
end
