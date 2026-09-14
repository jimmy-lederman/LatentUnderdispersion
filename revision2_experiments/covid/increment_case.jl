# Does the case study survive without Y_{i,t-1} in the rate?
#
#   julia -t 8 --gcthreads=1 --project=. revision2_experiments/covid/increment_case.jl [nsweeps] [ncounties]
#
# Fits the thinning (paper) sampler on a county subset four ways -- increments as data
# (use_offset = false) and cumulative counts with the offset, each with D inferred and
# with D pinned at 1 -- and reports the posterior of D and the held-out one-step log
# predictive density log E_post p(Y_{i,t} | Y_{i,t-1}, theta, phi, D). The held-out cells
# are the same in all four fits (20% of counties, last 10 days), so the increment and
# cumulative fits are scored on the same events. The case study needs D > 1 to beat D = 1;
# on increments, that requires the shared daily factors to absorb the reporting cycle that
# makes raw increments over-dispersed (var/mean ~ 6 in 7-day windows).
ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra, CSV, DataFrames
using MCMCDiagnosticTools

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
ncty    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
K, Q, Dmax, THIN = 5, 25, 9, 8

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty); cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1)); Ycum = Int.(cumdf[:, 2:end]); N, M = size(Ycum)
inc = Matrix{Int}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Ycum[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Ycum[i,t] - Ycum[i,t-1], 0); end
end
Random.seed!(101)
missing_counties = rand(N) .< 0.2
mask_NM = falses(N, M); for t in (M-9):M; mask_NM[:, t] = missing_counties; end
heldout = findall(mask_NM)
@printf("N=%d M=%d, held-out cells %d; increments: mean %.2f, zero %.1f%%\n", N, M, length(heldout), mean(inc), 100*mean(inc .== 0))

function fitone(useoff::Bool, dfix::Int, seed::Int)
    model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                   false, true, false, useoff, false)
    data = Dict("Y_NM" => useoff ? Ycum : inc)
    info = Dict("Y0_N" => useoff ? Y0_N : zeros(Int, N, 1))
    Random.seed!(seed)
    st = sample_prior(model, info)
    target = max(mean(inc) - st["eps"], 1e-6); cur = mean(st["U_NK"] * st["V_KM"])
    st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
    lik = zeros(length(heldout)); nk = 0
    Dsum = 0.0; Dceil = 0.0; Done = 0.0; mus = Vector{Vector{Float64}}()
    Random.seed!(7); cells = [(rand(1:N), rand(1:M)) for _ in 1:150]
    for s in 1:nsweeps
        (_d, st) = dfix > 0 ? backward_sample(model, data, st, mask_NM; skipupdate=["D_NM"]) :
                              backward_sample(model, data, st, mask_NM)
        dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
        if s > nsweeps ÷ 2 && s % THIN == 0
            nk += 1
            for (j, ci) in enumerate(heldout)
                lik[j] += exp(evalulateLogLikelihood(model, st, data, info, ci[1], ci[2]))
            end
            D = st["D_NM"]; Dsum += mean(D); Dceil += mean(D .== Dmax); Done += mean(D .== 1)
            U = st["U_NK"]; V = st["V_KM"]
            push!(mus, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
        end
    end
    lpd = mean(log.(lik ./ nk))
    return (lpd = lpd, Dmean = Dsum/nk, Dceil = Dceil/nk, Done = Done/nk, mus = mus)
end

results = Dict()
for (useoff, lab) in ((false, "INCREMENTS (no Y_{t-1})"), (true, "CUMULATIVE (offset)"))
    println("\n=== ", lab, " ===")
    t0 = time()
    r1 = fitone(useoff, 1, 11)
    @printf("  D = 1 fixed:   held-out lpd %8.4f   (%.1f min)\n", r1.lpd, (time()-t0)/60); flush(stdout)
    ra = fitone(useoff, 0, 21); rb = fitone(useoff, 0, 22)
    A = Array{Float64}(undef, length(ra.mus), 2, 150)
    for s in 1:length(ra.mus), p in 1:150; A[s,1,p] = ra.mus[s][p]; A[s,2,p] = rb.mus[s][p]; end
    rs = [rhat(reshape(A[:,:,p], size(A,1), 2, 1))[1] for p in 1:150]
    @printf("  D inferred:    held-out lpd %8.4f / %8.4f (two chains)   mean D %.2f / %.2f   at Dmax %.0f%%   at 1 %.0f%%   mu Rhat med/max %.2f/%.2f   (%.1f min)\n",
            ra.lpd, rb.lpd, ra.Dmean, rb.Dmean, 100*ra.Dceil, 100*ra.Done, median(rs), maximum(rs), (time()-t0)/60)
    @printf("  GAIN of inferred D over D = 1: %+.4f nats per held-out cell\n", (ra.lpd + rb.lpd)/2 - r1.lpd); flush(stdout)
    results[lab] = (r1, ra, rb)
end
