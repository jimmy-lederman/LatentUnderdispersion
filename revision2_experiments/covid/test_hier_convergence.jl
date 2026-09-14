# Does the hierarchical county-factor rate converge on the real Section 7.2 data?
#
#   julia -t 8 --project=. revision2_experiments/covid/test_hier_convergence.jl [nsweeps] [nchains] [ncounties]
#
# ncounties subsets the counties (default: all 3104). The identifiability question is
# structural -- theta and phi enter only as a product whatever N is -- so a subset answers
# it at a fraction of the cost. Note that converged sweeps on the full data are several
# times more expensive than warm-up sweeps, because the sampler's cost grows with the
# inferred D and D climbs during burn-in; a full-data 4-chain run of both models is hours,
# not minutes.
#
# theta_{i,k} and phi_{k,t} enter the model only through the product theta'phi, and phi's
# AR(1) prior is a drifting random walk that does not anchor its own scale. Freeing the
# rate b of theta therefore adds a third parameter along an already-weak direction. If
# that ridge is real, it shows up as: the PRODUCT mu_{i,t} = (theta'phi)_{i,t} converging
# across dispersed chains while b and the raw theta scale do not.
#
# So the diagnostics are split into two groups and reported separately:
#   IDENTIFIED     mu on fixed cells, eps, mean D  -- must converge for the model to be usable
#   SCALE-ONLY     b, mean(theta), mean(phi)       -- may not, and that is the diagnosis
#
# The flat model is run alongside as the control, from the same seeds.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra
using MCMCDiagnosticTools

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 800
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
nkeep   = nsweeps ÷ 4                     # last quarter retained
println("Julia ", VERSION, "  threads = ", Threads.nthreads())
println("$nchains chains x $nsweeps sweeps, keeping the last $nkeep")

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
cumdf = cumdf[2:end, 4:end]
if length(ARGS) >= 3
    ncty = parse(Int, ARGS[3])
    # every k-th county, so the subset spans the whole range of county sizes rather than
    # the first few alphabetically
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:, 1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM)
N, M = size(Y_NM)
info = Dict("Y0_N" => Y0_N)
K, Q, Dmax = 5, 25, 9
println("N = $N, M = $M")

Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:200]

mk(hier) = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                  0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, hier)

# One chain, returning per-draw functionals rather than states (a state is ~40 MB here).
function run_chain(model, data, info, seed, nsweeps, nkeep, cells)
    Random.seed!(seed)
    state = sample_prior(model, info)
    out = Dict(k => Float64[] for k in
               ("epsv", "Dmean", "b", "Umean", "Vmean", "loglikish"))
    mus = Vector{Vector{Float64}}()
    t0 = time()
    for s in 1:nsweeps
        (_d, state) = backward_sample(model, data, state)
        if s > nsweeps - nkeep
            U = state["U_NK"]::Matrix{Float64}; V = state["V_KM"]::Matrix{Float64}
            push!(out["epsv"],  state["eps"])
            push!(out["Dmean"], mean(state["D_NM"]))
            push!(out["b"],     model.hier_b ? state["b_mu"] : model.b)
            push!(out["Umean"], mean(U))
            push!(out["Vmean"], mean(V))
            push!(mus, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
            # a cheap stand-in for the joint log-density: total fitted mass
            push!(out["loglikish"], sum(U * V))
        end
        if s % 100 == 0
            @printf("    seed %d: sweep %d/%d  (%.1f min elapsed)\n", seed, s, nsweeps, (time()-t0)/60)
            flush(stdout)
        end
    end
    return out, mus
end

function diagnose(label, hier, nsweeps, nchains, nkeep, cells)
    model = mk(hier)
    println("\n", "="^80)
    println("$label  (hier_b = $hier)")
    println("="^80)
    outs = Vector{Any}(undef, nchains); muss = Vector{Any}(undef, nchains)
    for c in 1:nchains
        outs[c], muss[c] = run_chain(model, data, info, 1000 + c, nsweeps, nkeep, cells)
    end

    scalar(key) = reduce(hcat, [outs[c][key] for c in 1:nchains])         # (draws, chains)
    r3(A) = reshape(A, size(A,1), size(A,2), 1)
    MU = Array{Float64}(undef, nkeep, nchains, length(cells))
    for c in 1:nchains, s in 1:nkeep, p in eachindex(cells)
        MU[s, c, p] = muss[c][s][p]
    end

    safe(f, x) = try f(x) catch; [NaN] end
    @printf("\n%-12s %-10s %10s %10s %12s\n", "group", "quantity", "Rhat", "ESS_bulk", "chain means")
    println("-"^72)
    for (grp, key) in (("IDENTIFIED","epsv"), ("IDENTIFIED","Dmean"),
                       ("SCALE-ONLY","b"), ("SCALE-ONLY","Umean"), ("SCALE-ONLY","Vmean"),
                       ("IDENTIFIED","loglikish"))
        A = r3(scalar(key))
        cm = [mean(outs[c][key]) for c in 1:nchains]
        @printf("%-12s %-10s %10.3f %10.0f   %s\n", grp, key,
                safe(rhat, A)[1], safe(y -> ess(y; kind=:bulk), A)[1],
                join([@sprintf("%.4g", v) for v in cm], " "))
    end
    rmu = safe(rhat, MU); emu = safe(y -> ess(y; kind=:bulk), MU)
    @printf("%-12s %-10s %10.3f %10.0f   (max Rhat / mean ESS over %d cells)\n",
            "IDENTIFIED", "mu cells", maximum(filter(isfinite, rmu)),
            mean(filter(isfinite, emu)), length(cells))
    return nothing
end

diagnose("FLAT  theta ~ Gamma(1, 1)", false, nsweeps, nchains, nkeep, cells)
diagnose("HIER  theta ~ Gamma(1, b), b ~ Gamma(1,1)", true, nsweeps, nchains, nkeep, cells)

println("\nReading: if the HIER run shows Rhat ~ 1 on 'mu cells' but a large Rhat on 'b'")
println("and 'Umean', the ridge is real -- the fitted means are fine but the factor scale")
println("is not identified, and b should not be reported as a parameter estimate.")
