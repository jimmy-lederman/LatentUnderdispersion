# Validation gate for the two non-conjugate samplers in STARMFNNsparse.jl.
#
# Both new samplers are checked against an INDEPENDENT exact reference rather
# than against each other, so agreement cannot come from a shared bug. The
# reference is rejection sampling, which is exact by construction:
#
#   U conditional on (0,s):  p(t) prop. exp(-A t^2/2 + B t) t^(a-1) (s-t)^(a-1)
#     propose t ~ s * Beta(a,a)  [the exact power factor]
#     accept  w.p. exp(-A t^2/2 + Bt) / sup, sup at t = clamp(B/A, 0, s)
#
#   V conditional on (0,Inf): p(t) prop. exp(-A t^2/2 + B t) t^(c-1)
#     propose t ~ Gamma(c, rate = lam)  [power factor times e^{-lam t}]
#     accept  w.p. exp((B+lam) t - A t^2/2) / sup, sup at t = (B+lam)/A
#
# Rejection is far too slow for production -- that is precisely the point of the
# experiment -- but it is the right thing to validate against.
#
# Comparison is a two-sample Kolmogorov-Smirnov statistic against the 5% critical
# value. Reported per (a or c) setting, over a spread of (A, B) representing weak
# through strongly informative likelihoods.
#
# BURN-IN AND THINNING ARE NOT OPTIONAL HERE. The KS critical value assumes
# independent samples; MH and slice produce autocorrelated chains. A first pass
# without them flagged both samplers as wrong at every setting, which was an
# artifact: at a = 0.1 the MH chain accepts ~14% of proposals, so it sticks and
# the start value dominates. The chains are therefore burned in and thinned to
# near-independence before the test is applied -- the same trap that produced
# false Geweke failures earlier in this project.
#
# Usage: julia --project=../../.. validate_samplers.jl [nsamp]

include("../../../models/STARmodels/revision2/STARMFNNsparse.jl")
using Random, Printf, Statistics

nsamp = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20000

# ---------------------------------------------------------------- references
function ref_U(A, B, s, a, n; rng=Random.default_rng())
    tstar = A > 0 ? clamp(B / A, 0.0, s) : (B > 0 ? s : 0.0)
    logM = -A * tstar^2 / 2 + B * tstar
    out = Float64[]; sizehint!(out, n)
    while length(out) < n
        t = s * rand(rng, Beta(a, a))
        (t <= 0 || t >= s) && continue
        if log(rand(rng)) < (-A * t^2 / 2 + B * t) - logM
            push!(out, t)
        end
    end
    return out
end

function ref_V(A, B, c, n; rng=Random.default_rng())
    lam = max(1.0, -B)                     # gamma rate for the proposal
    Bt = B + lam                           # residual linear term after tilting
    tstar = A > 0 ? max(Bt / A, 0.0) : 0.0
    logM = -A * tstar^2 / 2 + Bt * tstar
    out = Float64[]; sizehint!(out, n)
    while length(out) < n
        t = rand(rng, Gamma(c, 1 / lam))
        t <= 0 && continue
        if log(rand(rng)) < (-A * t^2 / 2 + Bt * t) - logM
            push!(out, t)
        end
    end
    return out
end

# ---------------------------------------------------------------- samplers under test
BURN = 20_000

function mh_U(A, B, s, a, n, THIN)
    t = s / 2; out = zeros(n); k = 0
    for i in 1:(BURN + n * THIN)
        p = truncgauss(A, B, 0.0, s)
        lr = (a - 1) * (log(p) + log(s - p) - log(t) - log(s - t))
        isfinite(lr) && log(rand()) < lr && (t = p)
        if i > BURN && (i - BURN) % THIN == 0; out[k += 1] = t; end
    end
    return out
end
function slice_U(A, B, s, a, n, THIN)
    t = s / 2; out = zeros(n); k = 0
    f(x) = -A * x^2 / 2 + B * x + (a - 1) * (log(x) + log(s - x))
    for i in 1:(BURN + n * THIN)
        t = slice_bounded(f, clamp(t, 1e-12 * s, s - 1e-12 * s), 0.0, s)
        if i > BURN && (i - BURN) % THIN == 0; out[k += 1] = t; end
    end
    return out
end
function mh_V(A, B, c, n, THIN)
    t = 1.0; out = zeros(n); k = 0
    for i in 1:(BURN + n * THIN)
        p = truncgauss(A, B, 0.0, Inf)
        lr = (c - 1) * (log(p) - log(t))
        isfinite(lr) && log(rand()) < lr && (t = p)
        if i > BURN && (i - BURN) % THIN == 0; out[k += 1] = t; end
    end
    return out
end
function slice_V(A, B, c, n, THIN)
    t = 1.0; out = zeros(n); k = 0
    f(x) = -A * x^2 / 2 + B * x + (c - 1) * log(x)
    w = A > 0 ? max(sqrt(1 / A), 1e-8) : 1.0
    for i in 1:(BURN + n * THIN)
        t = slice_positive(f, max(t, 1e-12), w)
        if i > BURN && (i - BURN) % THIN == 0; out[k += 1] = t; end
    end
    return out
end

"Two-sample KS statistic and the 5% critical value."
function ks(x, y)
    xs, ys = sort(x), sort(y)
    nx, ny = length(xs), length(ys)
    i = j = 1; d = 0.0
    while i <= nx && j <= ny
        if xs[i] <= ys[j]; i += 1 else j += 1 end
        d = max(d, abs(i / nx - j / ny))
    end
    return d, 1.36 * sqrt((nx + ny) / (nx * ny))
end

"Smallest thinning interval at which the chain is KS-indistinguishable from
exact draws. This IS the cost measurement: a correct sampler always passes
eventually, so what varies is how many iterations one effective draw costs."
function min_thin(chain, ref, n, thins)
    for th in thins
        d, c = ks(chain(n, th), ref)
        d <= c && return th, d, c
    end
    return -1, NaN, NaN
end
const THINS = [10, 50, 250, 1000, 5000, 20000]

Random.seed!(20260901)
# (A, B) spanning weak to strongly informative conditionals
CASES = [(0.5, 0.2), (5.0, 1.0), (50.0, -3.0), (200.0, 10.0)]

println("Thinning needed to be KS-indistinguishable from exact draws (-1 = fails at 20000)")
println("U conditional on (0, s), s = 1.0")
@printf("%-6s %-13s %10s %10s\n", "a", "(A,B)", "mh", "slice")
for a in (0.5, 0.25, 0.1), (A, B) in CASES
    r = ref_U(A, B, 1.0, a, nsamp)
    t1, _, _ = min_thin((n, th) -> mh_U(A, B, 1.0, a, n, th), r, nsamp, THINS)
    t2, _, _ = min_thin((n, th) -> slice_U(A, B, 1.0, a, n, th), r, nsamp, THINS)
    @printf("%-6.2f %-13s %10d %10d\n", a, "($A,$B)", t1, t2)
end

println("\nV conditional on (0, Inf)")
@printf("%-6s %-13s %10s %10s\n", "c", "(A,B)", "mh", "slice")
for c in (0.5, 0.25, 0.1), (A, B) in CASES
    r = ref_V(A, B, c, nsamp)
    t1, _, _ = min_thin((n, th) -> mh_V(A, B, c, n, th), r, nsamp, THINS)
    t2, _, _ = min_thin((n, th) -> slice_V(A, B, c, n, th), r, nsamp, THINS)
    @printf("%-6.2f %-13s %10d %10d\n", c, "($A,$B)", t1, t2)
end
