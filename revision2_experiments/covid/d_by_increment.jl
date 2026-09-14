# Which cells drive D to its ceiling?
#
#   julia -t 6 --project=. revision2_experiments/covid/d_by_increment.jl [nsweeps] [ncounties] [Dmax]
#
# The Dmax ladder showed D pinned at whatever Dmax is set to (88% of cells at Dmax = 9,
# 82% at Dmax = 15), so the reported order reflects the bound rather than the data. This
# asks WHICH cells are doing it, and the hypothesis is structural rather than numerical:
#
#   36% of county-days have an increment of EXACTLY ZERO, and 54% are <= 2.
#
# For a zero-increment cell, Y_t = Y_{t-1} while the rate is mu = Y_{t-1} + eps + growth.
# When eps + growth < 1 the MEDIAN of Poisson(mu) is exactly Y_{t-1}, so concentrating
# harder -- a larger D -- puts MORE mass on the observed value, not less. Those cells want
# D unbounded. (This also explains eps collapsing towards 0 on real data: pushing eps down
# keeps mu just above Y_{t-1}, so the median lands exactly on the observation.)
#
# If the hypothesis holds, D at Dmax should be concentrated in the zero- and tiny-increment
# cells, and cells with substantial increments should sit at moderate D. If instead every
# bucket is pinned at Dmax, the story is different and the bound is binding for all cells.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8000
ncty    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
Dmax    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 9
println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  Dmax = ", Dmax)

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
K, Q = 5, 25
println("N = $N, M = $M, $nsweeps sweeps")

incI = Matrix{Int}(undef, N, M)
for i in 1:N
    incI[i,1] = Y_NM[i,1] - Y0_N[i]
    for t in 2:M; incI[i,t] = Y_NM[i,t] - Y_NM[i,t-1]; end
end
inc = max.(Float64.(incI), 0.0)

model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true)
Random.seed!(4242)
st = sample_prior(model, info)
target = max(mean(inc) - st["eps"], 1e-6)
cur = mean(st["U_NK"] * st["V_KM"])
st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))

# accumulate the posterior mean of D per cell over the retained half
Dsum = zeros(Float64, N, M); nk = 0
t0 = time()
for s in 1:nsweeps
    global st, nk
    (_d, st) = backward_sample(model, data, st)
    if s > nsweeps ÷ 2
        Dsum .+= st["D_NM"]; nk += 1
    end
    s % 1000 == 0 && (@printf("  sweep %d/%d (%.1f min)  meanD %.3f  eps %.4f\n",
                              s, nsweeps, (time()-t0)/60, mean(st["D_NM"]), st["eps"]); flush(stdout))
end
Dbar = Dsum ./ nk

buckets = [("inc < 0 (corrections)", v -> v < 0),
           ("inc == 0",              v -> v == 0),
           ("inc == 1",              v -> v == 1),
           ("inc 2-5",               v -> 2 <= v <= 5),
           ("inc 6-20",              v -> 6 <= v <= 20),
           ("inc 21-100",            v -> 21 <= v <= 100),
           ("inc > 100",             v -> v > 100)]

println("\nposterior mean D by observed increment  (Dmax = $Dmax)")
@printf("%-24s %8s %10s %12s %14s\n", "bucket", "cells", "share", "mean Dbar", "frac near Dmax")
println("-"^72)
for (name, f) in buckets
    idx = findall(f, vec(incI))
    isempty(idx) && continue
    d = vec(Dbar)[idx]
    @printf("%-24s %8d %9.1f%% %12.3f %11.1f%%\n",
            name, length(idx), 100*length(idx)/length(incI), mean(d),
            100*count(>=(Dmax - 0.5), d)/length(d))
end

allD = vec(Dbar)
@printf("\noverall mean Dbar = %.3f, frac within 0.5 of Dmax = %.1f%%\n",
        mean(allD), 100*count(>=(Dmax - 0.5), allD)/length(allD))
println("\nIf the zero/tiny-increment buckets are the ones pinned at Dmax, the ceiling is")
println("structural: those cells want an unbounded order because the median of the rate")
println("sits exactly on the observation. Dmax then sets the answer, not the data.")
