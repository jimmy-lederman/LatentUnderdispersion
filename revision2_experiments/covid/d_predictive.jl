# Does the D machinery earn its place, and does the Dmax ceiling cost prediction?
#
#   julia -t 4 --project=. revision2_experiments/covid/d_predictive.jl [nsweeps] [nchains] [ncounties]
#
# D is a computational device in Section 7.2 rather than an estimand: it buys a tighter
# conditional and therefore better forecasts. Under that framing D does not need to mix,
# and its ESS is not a quantity anyone reports. What DOES need to hold is that the
# predictive claim is stable, and that the two things the Dmax ladder called into question
# are answered on predictive grounds:
#
#   1. 88% of cells sit AT Dmax, so for most cells the fitted model is effectively fixed-D.
#      Does D INFERRED actually beat D FIXED at the same value on held-out density? If not,
#      the OddBinomial / Polya-gamma apparatus of Eq. (13) is not earning its keep, and
#      simplifying to a fixed D is a stronger paper, not a weaker one.
#
#   2. Does raising the ceiling improve prediction? If held-out density plateaus by Dmax = 9,
#      the bound is harmless and "9 was sufficient" becomes an empirical statement rather
#      than an unexamined default. If Dmax = 15 predicts better, the choice needs revisiting.
#
# Held-out density is an aggregate identified quantity and it converges far better than
# per-cell mu (aggregate growth plateaued by ~1500 sweeps while per-cell Rhat stayed ~2.4),
# so it is the right thing to hang the claim on.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, LogExpFunctions

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200
nkeep   = 500
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

# same forecasting mask as the run script: last 10 days for a random 20% of counties
Random.seed!(101)
missing_counties = rand(N) .< 0.2
mask_NM = falses(N, M)
for t in (M-9):M; mask_NM[:, t] = missing_counties; end
heldout = findall(mask_NM)
println("N = $N, M = $M, held-out cells = $(length(heldout))")
println("$nchains chains x $nsweeps sweeps, keeping $nkeep\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)

function run_arm(label, Dfix, Dmax)
    model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                   0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true)
    keep = Vector{Any}()
    tt = 0.0
    for c in 1:nchains
        Random.seed!(700 + c)
        st = sample_prior(model, info)
        target = max(mean_inc - st["eps"], 1e-6)
        cur = mean(st["U_NK"] * st["V_KM"])
        st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
        Dfix > 0 && (st["D_NM"] = fill(Dfix, N, M))
        thin = max(1, (nsweeps ÷ 2) ÷ nkeep)
        tt += @elapsed for s in 1:nsweeps
            (_d, st) = Dfix > 0 ?
                backward_sample(model, data, st, mask_NM; skipupdate=["D_NM"]) :
                backward_sample(model, data, st, mask_NM)
            Dfix > 0 && (st["D_NM"] = fill(Dfix, N, M))
            if s > nsweeps ÷ 2 && (s % thin == 0)
                push!(keep, Dict("U_NK"=>copy(st["U_NK"]), "V_KM"=>copy(st["V_KM"]),
                                 "eps"=>st["eps"], "D_NM"=>copy(st["D_NM"]),
                                 "Y0_N"=>st["Y0_N"]))
            end
        end
    end
    S = length(keep)
    ll = Matrix{Float64}(undef, S, length(heldout))
    Threads.@threads for h in eachindex(heldout)
        ci = heldout[h]
        for s in 1:S
            ll[s,h] = evalulateLogLikelihood(model, keep[s], data, info, ci[1], ci[2])
        end
    end
    percell = [logsumexp(ll[:,h]) - log(S) for h in eachindex(heldout)]
    ir = mean(percell)
    md = mean(mean(k["D_NM"]) for k in keep)
    @printf("%-34s  inforate %9.4f   mean D %6.3f   %6.1f min  (%d draws)\n",
            label, ir, md, tt/60, S)
    flush(stdout)
    return ir
end

@printf("%-34s  %9s   %6s\n", "arm", "inforate", "mean D")
println("-"^80)
ir1  = run_arm("D = 1 fixed (Poisson baseline)", 1, 9)
ir9  = run_arm("D = 9 fixed",                     9, 9)
irI9 = run_arm("D inferred, Dmax = 9",            0, 9)
irI15= run_arm("D inferred, Dmax = 15",           0, 15)

println()
@printf("gain over Poisson  D=9 fixed        : %+.4f\n", ir9 - ir1)
@printf("gain over Poisson  D inferred (9)   : %+.4f\n", irI9 - ir1)
@printf("gain over Poisson  D inferred (15)  : %+.4f\n", irI15 - ir1)
@printf("inferred(9) - fixed(9)              : %+.4f   <- does heterogeneity earn its keep?\n", irI9 - ir9)
@printf("inferred(15) - inferred(9)          : %+.4f   <- does raising the ceiling help?\n", irI15 - irI9)
