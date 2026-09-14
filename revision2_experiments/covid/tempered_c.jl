# Parallel tempering on the phi smoothness coupling c, for the Section 7.2 sampler.
#
#   julia -t 8 --gcthreads=1 --project=. revision2_experiments/covid/tempered_c.jl [nsweeps] [nsystems] [ncounties]
#
# WHY. The pairwise shear move showed that the c = 100 gamma-Markov prior on phi rejects every
# mu-preserving mixing of two factors by 50-400 nats. So the factorisation is identified BY
# that prior, and the landscape over factorisations is rugged: sharp local modes (distinct
# smooth 5-row splits of the same growth surface) separated by rough valleys. Chains fall into
# different modes and no mu-preserving move can carry them across. Weakening c to 10 helps
# only 10-18%: at c = 10 the shear penalty is ~20 nats, still prohibitive.
#
# Tempering is the standard answer for a rugged landscape, and here it is unusually cheap:
# the target at temperature c differs from the target at c = 100 ONLY in the phi prior. The
# likelihood, the theta / D / beta / tau priors, and the augmentation are identical. So a
# swap between the chains at c_a and c_b is accepted with probability
#     min(1, p_a(phi_b) p_b(phi_a) / (p_a(phi_a) p_b(phi_b)))
# where p_c is the gamma-Markov prior at coupling c -- four evaluations of an O(K*M) function.
# Each temperature runs the ordinary Gibbs sweep at its own c. Everything is still data
# augmentation; nothing is gradient-based.
#
# Diagnostics are computed on the c = 100 chain of each system only -- that is the target.
# The auxiliary chains exist to be swapped from.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra
using MCMCDiagnosticTools, Combinatorics

nsweeps  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
nsystems = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4      # independent tempered systems, for Rhat
ncty     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
const TEMPS = [100.0, 20.0, 4.0]                              # c ladder; TEMPS[1] is the target
const THIN  = 8
println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  c ladder = ", TEMPS)

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
step = max(1, size(cumdf,1) ÷ ncty); cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
Y0_N = Int.(reshape(cumdf[:,1], :, 1)); Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
K, Q, Dmax = 5, 25, 9
Random.seed!(101); mc = rand(N) .< 0.2
mask_NM = falses(N, M); for t in (M-9):M; mask_NM[:, t] = mc; end
heldout = findall(mask_NM)
Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]; icty = sort(randperm(N)[1:60])
inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
println("N = $N, M = $M, $nsystems systems x $(length(TEMPS)) temperatures x $nsweeps sweeps\n")

mk(c) = covid4(N, M, K, Q, Dmax, 1.0, 1.0, c, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true, false, true, false)
models = [mk(c) for c in TEMPS]

# log phi-prior of a whole state under the model at temperature index a
function lp_phi_state(a, st)
    V = st["V_KM"]::Matrix{Float64}
    s = 0.0
    for k in 1:K; s += logprior_phi_vec(models[a], view(V, k, :)); end
    return s
end

function init_state(seed)
    Random.seed!(seed)
    st = sample_prior(models[1], info)
    st["V_KM"] = st["V_KM"] .* (max(mean_inc - st["eps"], 1e-6) / max(mean(st["U_NK"]*st["V_KM"]), 1e-12))
    return st
end

function run_system(seed)
    T = length(TEMPS)
    states = [init_state(seed + 100*(a-1)) for a in 1:T]
    ll = Float64[]; mu = Vector{Vector{Float64}}()
    th = Vector{Matrix{Float64}}(); ph = Vector{Matrix{Float64}}()
    nswap = zeros(Int, T-1); nacc = zeros(Int, T-1)
    t0 = time()
    for s in 1:nsweeps
        for a in 1:T
            (_d, states[a]) = backward_sample(models[a], data, states[a], mask_NM)
        end
        # adjacent swaps, alternating parity so every pair is tried every other sweep
        for a in (isodd(s) ? 1 : 2):2:(T-1)
            b = a + 1
            nswap[a] += 1
            la_a = lp_phi_state(a, states[a]); la_b = lp_phi_state(a, states[b])
            lb_a = lp_phi_state(b, states[a]); lb_b = lp_phi_state(b, states[b])
            if log(rand()) < (la_b + lb_a) - (la_a + lb_b)
                states[a], states[b] = states[b], states[a]
                nacc[a] += 1
            end
        end
        if s % THIN == 0
            st = states[1]      # the c = 100 chain
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            l = 0.0
            for ci in heldout; l += evalulateLogLikelihood(models[1], st, data, info, ci[1], ci[2]); end
            push!(ll, l); push!(mu, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
            sk = vec(sum(U, dims=1))
            push!(th, U[icty,:] ./ reshape(sk,1,K)); push!(ph, V .* reshape(sk,K,1))
        end
        if s % 1000 == 0
            @printf("    seed %d: %d/%d (%.1f min)  swap acc: %s\n", seed, s, nsweeps, (time()-t0)/60,
                    join([@sprintf("%.2f", nacc[a]/max(nswap[a],1)) for a in 1:T-1], " "))
            flush(stdout)
        end
    end
    return ll, mu, th, ph, nacc ./ max.(nswap, 1)
end

res = Vector{Any}(undef, nsystems)
for c in 1:nsystems
    res[c] = run_system(500 + c)
    @printf("  system %d done, swap acceptance %s\n", c, join([@sprintf("%.2f", x) for x in res[c][5]], " "))
    flush(stdout)
end

safe(f,x) = try f(x) catch; NaN end
r3(A) = reshape(A, size(A,1), size(A,2), 1)
function diag3(A)
    rs = filter(isfinite, [safe(_ -> rhat(r3(A[:,:,p]))[1], p) for p in 1:size(A,3)])
    e  = mean(filter(isfinite, [safe(x -> ess(r3(reshape(x,:,1)); kind=:bulk)[1], A[:,c,p])
                                for c in 1:size(A,2), p in 1:size(A,3)]))
    return (median(rs), quantile(rs, .9), maximum(rs)), e
end
function align_draws(thw, niter=8)
    perms = collect(permutations(1:K)); ref = mean(thw[1])
    asg = [[collect(1:K) for _ in eachindex(thw[c])] for c in 1:nsystems]
    for _ in 1:niter
        acc = zeros(size(ref)); n = 0
        for c in 1:nsystems, s in eachindex(thw[c])
            p = perms[argmin([sum(abs2, ref .- thw[c][s][:, pp]) for pp in perms])]
            asg[c][s] = p; acc .+= thw[c][s][:, p]; n += 1
        end
        ref = acc ./ n
    end
    asg
end

nk = length(res[1][1]); idx = (nk ÷ 2 + 1):nk          # second half of retained draws
nkw = length(idx)
L  = Array{Float64}(undef, nkw, nsystems, 1)
MU = Array{Float64}(undef, nkw, nsystems, length(cells))
for c in 1:nsystems; L[:,c,1] = res[c][1][idx]; for s in 1:nkw, p in eachindex(cells); MU[s,c,p] = res[c][2][idx[s]][p]; end; end
thw = [[res[c][3][i] for i in idx] for c in 1:nsystems]; phw = [[res[c][4][i] for i in idx] for c in 1:nsystems]
asg = align_draws(thw)
TH = Array{Float64}(undef, nkw, nsystems, 60*K); PH = Array{Float64}(undef, nkw, nsystems, K*M)
COM = Array{Float64}(undef, nkw, nsystems, K); SHR = Array{Float64}(undef, nkw, nsystems, K)
for c in 1:nsystems, s in 1:nkw
    p = asg[c][s]; TH[s,c,:] = vec(thw[c][s][:, p]); P = phw[c][s][p, :]; PH[s,c,:] = vec(P)
    tot = vec(sum(P, dims=2))
    for k in 1:K; COM[s,c,k] = sum(t*P[k,t] for t in 1:M) / max(tot[k], 1e-300); end
    SHR[s,c,:] = tot ./ max(sum(tot), 1e-300)
end
println()
@printf("%-26s %19s %10s   (c = 100 chain, sweeps %d-%d)\n", "quantity", "Rhat med/p90/max", "ESS/chain", nsweeps÷2, nsweeps)
println("-"^70)
for (nm, A) in (("held-out log density", L), ("mu (150 cells)", MU), ("theta/s_k aligned", TH),
                ("s_k*phi aligned", PH), ("FACTOR centre of mass", COM), ("FACTOR activity share", SHR))
    r, e = diag3(A); @printf("%-26s %5.2f/%5.2f/%5.2f %10.1f\n", nm, r[1], r[2], r[3], e)
end
println("\nCompare against the untempered c = 100 run at the same size (4000-6000 window):")
println("  mu 1.41  theta 1.65  phi 2.27  centre 3.30  share 3.42")
