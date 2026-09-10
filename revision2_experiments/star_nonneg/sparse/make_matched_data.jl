# Well-specified sparse data: for each concentration a, generate from EXACTLY the
# model's own prior, then fit with that same a. No misspecification anywhere.
#
#   U_:k ~ Dirichlet(a * 1_N),  V_kj ~ Gamma(a, rate),  Y ~ CMP(mu, nu = 2)
#
# COUNT LEVEL IS HELD FIXED ACROSS a. E[V] = a/rate, so lowering the shape also
# lowers the prior mean of mu unless the rate is moved with it; rate = a/VTARGET
# keeps E[mu] = K/N * VTARGET constant. Without this, sparser settings would also
# be lower-count settings, and a separate test already showed that sparser truth
# means more zeros, less information, and worse convergence FOR EVERY MODEL
# (support 4 -> 2 -> 1 pushed R-hat 1.75 -> 2.06 -> 2.40). Holding counts fixed
# keeps the experiment about sparsity rather than about information volume.
#
# Usage: julia --project=../../.. make_matched_data.jl [a ...]

using Random, Distributions, Printf, CSV, DataFrames, SpecialFunctions, Statistics

const N, M, K = 20, 20, 3
const NU = 2.0
const VTARGET = parse(Float64, get(ENV, "VTARGET", "40.0"))  # sets the count level
const SEED = parse(Int, get(ENV, "SEED", "20260910"))

function rand_cmp(rng, lambda, nu)
    lambda <= 0 && return 0
    ymax = max(50, ceil(Int, 12 * lambda^(1 / nu)))
    lp = [y * log(lambda) - nu * loggamma(y + 1.0) for y in 0:ymax]
    lp .-= maximum(lp); p = exp.(lp); p ./= sum(p)
    u, c = rand(rng), 0.0
    for y in 0:ymax; c += p[y + 1]; u <= c && return y; end
    return ymax
end

acs = isempty(ARGS) ? [1.0, 0.5, 0.25, 0.1, 0.05, 0.01] : parse.(Float64, ARGS)
d = joinpath(@__DIR__, "data"); mkpath(d)
wr(path, A) = CSV.write(path, DataFrame(hcat(string.(1:size(A,1)), A),
                                        ["", ["V$(j)" for j in 1:size(A,2)]...]))
@printf("%-7s %8s %8s %8s %9s %11s\n", "a", "rate", "mean mu", "mean Y", "% zero", "U<0.01")
for a in acs
    rng = MersenneTwister(SEED)
    rate = a / VTARGET                      # keeps E[V] = VTARGET for every a
    U = rand(rng, Dirichlet(fill(a, N)), K)
    V = rand(rng, Gamma(a, 1 / rate), K, M)
    Mu = U * V
    Y = [rand_cmp(rng, Mu[i, j], NU) for i in 1:N, j in 1:M]
    tag = "Matched$(a)"
    wr(joinpath(d, tag * "factor.csv"), Y)
    wr(joinpath(d, tag * "U_NK.csv"), permutedims(U))
    @printf("%-7.2f %8.4f %8.2f %8.2f %8.0f%% %10.0f%%\n",
            a, rate, mean(Mu), mean(Y), 100*mean(Y .== 0), 100*mean(U .< 0.01))
end
