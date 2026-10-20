# Per-sweep cost of each arm of the information-gain grid, at the campaign's settings.
#
#   julia -t 6 --project=. revision2_experiments/covid/arm_cost.jl [warm] [timed]
#
# The grid runs 6 arms at a COMMON sweep count, but the cost per sweep is not common: the
# latent-sum draw is O(D) per cell, and a pinned D additionally skips the whole D /
# Polya-gamma / beta / tau block (skipupdate). The figures previously used here mixed a
# Q = 25 measurement (~4.5 s/sweep at D ~ 8) with a Q = 1 one (the final run's 1.88-2.54
# s/sweep at mean D 7.9), which is not a consistent basis for sizing the campaign.
#
# Measured on the FULL matrix at the campaign's configuration. Absolute numbers are this
# machine's; the RATIOS carry to the cluster, and the inferred arm is anchored by the
# measured 471-636 min/chain of output/covid/diag_final_Q1.

ROOTDIR = abspath(joinpath(@__DIR__, "../.."))
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra

warm  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
timed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  BLAS = ", BLAS.get_num_threads())

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
Y0_N = Int.(reshape(cumdf[:,1], :, 1)); Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
Random.seed!(101); mc = rand(N) .< 0.2
mask_NM = falses(N, M); for t in (M-9):M; mask_NM[:, t] = mc; end
inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
println("N = $N, M = $M, warm = $warm, timed = $timed")

K, Q, Dmax, EPS = 5, 1, 9, 0.5
model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
               false, true, false, true, false, true, false, true, false)

function armcost(dfix)
    Random.seed!(1)
    st = sample_prior(model, info)
    st["eps"] = EPS
    dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
    target = max(mean_inc - EPS, 1e-6)
    st["V_KM"] = st["V_KM"] .* (target / max(mean(st["U_NK"] * st["V_KM"]), 1e-12))
    sweep!() = begin
        (_d, st2) = dfix > 0 ?
            backward_sample(model, data, st, mask_NM; skipupdate=["D_NM"]) :
            backward_sample(model, data, st, mask_NM)
        st = st2
        st["eps"] = EPS
        dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
    end
    for _ in 1:warm; sweep!(); end
    t0 = time(); for _ in 1:timed; sweep!(); end
    ((time() - t0)/timed, mean(st["D_NM"]))
end

println()
@printf("%-22s %12s %10s %10s\n", "arm", "s/sweep", "mean D", "rel to D=9")
res = Tuple{String,Float64,Float64}[]
for (label, dfix) in [("D = 1 (pinned)",1), ("D = 3 (pinned)",3), ("D = 5 (pinned)",5),
                      ("D = 7 (pinned)",7), ("D = 9 (pinned)",9), ("D inferred",0)]
    (sps, md) = armcost(dfix)
    push!(res, (label, sps, md))
    flush(stdout)
end
ref = res[5][2]
for (label, sps, md) in res
    @printf("%-22s %12.3f %10.3f %10.2fx\n", label, sps, md, sps/ref)
end
println("\nNOTE: the inferred arm is measured a few sweeps from the prior draw, where mean D")
println("is still near 5; at its converged D ~ 7.9 it costs more than shown here.")
