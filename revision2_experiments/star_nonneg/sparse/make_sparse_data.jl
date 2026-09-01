# Sparse-factor DGP for the non-conjugacy experiment.
#
# The Section 6.3 data uses Gamma shapes of 1 (base) and 100 (boost), which
# gives DENSE, blocky loadings. Here the same block layout is generated with
# concentrations BELOW one, so most loadings sit near zero and a few carry all
# the mass. Sparse priors of this kind are the regime where the choice of
# hyperparameter actually changes inference (cf. Wallach, Mimno & McCallum,
# "Rethinking LDA: Why Priors Matter", 2009).
#
# Why this DGP is the interesting one: fitting it well wants a Dirichlet
# concentration a < 1 and a gamma shape c < 1. The order statistic models take
# those for free -- theta | counts is Dirichlet(a + counts) and phi | counts is
# Gamma(c + counts, ...) whatever a and c are. STAR-NN's Gibbs updates are exact
# ONLY at a = c = 1 (see models/STARmodels/revision2/STARMFNN.jl): away from that
# point the mass-shift conditional picks up t^(a-1)(s-t)^(a-1) and the weight
# conditional picks up t^(c-1), neither of which is normal. Sparsity is the
# sharp case because those factors blow up at the boundary, exactly where a
# Gaussian proposal has no mass.
#
# Conventions match revison_experiments/factorexperiment/data: contiguous
# blocks, columns of U on the simplex, CSVs with an R-style index column, U and
# V stored K x N and K x M.
#
# Usage: julia --project=../../.. make_sparse_data.jl [seed]

using Random, Distributions, Printf, CSV, DataFrames, SpecialFunctions, Statistics

const N, M, K = 20, 20, 3
# Overridable from the environment so the sparsity level can be calibrated
# without editing the file: NU, A_HI, A_LO, C_HI, C_LO, RATE.
env(k, d) = parse(Float64, get(ENV, k, string(d)))
const NU = env("NU", 2.0)               # CMP exponent; > 1 is underdispersed
# Sparsity in U is imposed by an explicit SUPPORT rather than by a tiny
# Dirichlet concentration. A concentration below one over all 20 rows is
# degenerate: it puts 60-85% of a factor's mass on a single row, which is a
# spike, not a sparse factor. Instead each factor is active on SU rows inside
# its block, roughly evenly (Dirichlet(1) over the support), and negligible
# elsewhere -- the regime a sparse prior is actually designed for.
const SU = Int(env("SU", 4))            # active rows per factor
const A_ON, A_OFF = env("A_ON", 1.0), env("A_OFF", 0.002)
# V stays comparatively dense: sparse U times sparse V leaves mu at zero almost
# everywhere (80%+ empty), which is a degenerate matrix rather than a sparse one.
# The off-block weight supplies the background level.
const C_HI, C_LO = env("C_HI", 5.0), env("C_LO", 0.5)     # gamma shape in/out
const RATE = env("RATE", 0.045)         # gamma rate; matches the dense data's mu level

seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20260812
blockof(i, n, k) = min(k, div((i - 1) * k, n) + 1)

"Exact draw from CMP(lambda, nu) by inverse CDF on an adaptive support."
function rand_cmp(rng, lambda, nu)
    lambda <= 0 && return 0
    ymax = max(50, ceil(Int, 12 * lambda^(1 / nu)))
    lp = [y * log(lambda) - nu * loggamma(y + 1.0) for y in 0:ymax]
    lp .-= maximum(lp)
    p = exp.(lp); p ./= sum(p)
    u, c = rand(rng), 0.0
    for y in 0:ymax
        c += p[y + 1]
        u <= c && return y
    end
    return ymax
end

rng = MersenneTwister(seed)
U = zeros(N, K); V = zeros(K, M)
support = Dict{Int,Vector{Int}}()
for k in 1:K
    inblock = [i for i in 1:N if blockof(i, N, K) == k]
    support[k] = sort(shuffle(rng, inblock)[1:min(SU, length(inblock))])
    U[:, k] = rand(rng, Dirichlet([i in support[k] ? A_ON : A_OFF for i in 1:N]))
    for j in 1:M
        V[k, j] = rand(rng, Gamma(blockof(j, M, K) == k ? C_HI : C_LO, 1 / RATE))
    end
end
Mu = U * V
Y = [rand_cmp(rng, Mu[i, j], NU) for i in 1:N, j in 1:M]

@printf("seed %d | SU=%d a_on=%.2f a_off=%.3f c=(%.2f,%.2f) rate=%.3f nu=%.1f\n",
        seed, SU, A_ON, A_OFF, C_HI, C_LO, RATE, NU)
@printf("mu   : mean %8.2f  median %8.2f  max %9.2f\n", mean(Mu), median(Mu), maximum(Mu))
@printf("Y    : mean %8.2f  var/mean %6.3f  max %5d  zeros %4.1f%%\n",
        mean(Y), var(Y) / mean(Y), maximum(Y), 100 * mean(Y .== 0))
@printf("U    : frac < 0.01 = %4.1f%%   largest-in-column mean %.3f\n",
        100 * mean(U .< 0.01), mean(maximum(U, dims = 1)))
@printf("V    : frac < 1%% of column max = %4.1f%%\n",
        100 * mean(V .< 0.01 .* maximum(V, dims = 2)))
for k in 1:K
    @printf("  factor %d: support %s holds %.3f of the mass\n",
            k, string(support[k]), sum(U[support[k], k]))
end

if get(ENV, "WRITE", "0") == "1"
    d = joinpath(@__DIR__, "data")
    wr(path, A) = CSV.write(path, DataFrame(hcat(string.(1:size(A, 1)), A),
                                            ["", ["V$(j)" for j in 1:size(A, 2)]...]))
    wr(joinpath(d, "SparseCMPfactor.csv"), Y)
    wr(joinpath(d, "SparseCMPU_NK.csv"), permutedims(U))
    wr(joinpath(d, "SparseCMPV_KM.csv"), V)
    println("wrote data/SparseCMP{factor,U_NK,V_KM}.csv")
end
