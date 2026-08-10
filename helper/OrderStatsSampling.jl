using Distributions
using LogExpFunctions
using Random
using SpecialFunctions
# include("PoissonMedianFunctions.jl")

function safeTrunc(dist, lower, upper; n=1)
    # n == 0 returns an empty vector rather than the scalar 0, so the return type
    # is always Vector{eltype(dist)}. Call sites either sum (sum of empty == 0,
    # as before) or append! under a count guard, so behaviour is unchanged.
    if lower == 0 && upper == Inf
        return rand(dist, n)
    end
    # Feasibility is decided by the MASS of [lower, upper], not by the pdf at an
    # endpoint. The previous endpoint test failed whenever the interval holds
    # essentially all the mass but its far endpoint sits deep in the tail
    # (e.g. Poisson(2) truncated to [0, 199]): it declared the interval
    # infeasible and returned the endpoint as a constant, inflating the latent
    # values. randTruncDirect handles the same decision correctly.
    return randTruncDirect(dist, lower, upper, n)
end

function saferand(dist, n, y)
    if n == 0
        return 0
    end
    try
        return rand(dist, n)
    catch ex
        for i in 1:100
            try
                return rand(dist,n)
            catch ex
                # do nothing
            end
        end
        return y
    end
end


# Per-(dist, Y) quantities that are constant across the sequential classification steps.
# Hoisting them is an algebraic no-op: every downstream formula consumes the exact same
# floating-point values it previously recomputed, so results are bitwise identical.
struct OSConsts
    pdfY::Float64
    lpY::Float64
    FY::Float64
    FYm1::Float64
    lcdfYm1::Float64
    lccdfY::Float64
end

# StatsFuns' Poisson logcdf/logccdf go through an incomplete gamma that THROWS
# ("Unsupported order |nu| > 50 off the positive real axis") for some (mu, Y) with Y > 50 --
# e.g. Poisson(1000) at Y=60, or Poisson(1e4) at Y=100, while Poisson(1000) at Y=16 and
# Poisson(500) at Y=51 are fine. Flights has Y from 16 to 368, so any large mu drawn at
# initialisation can land in that zone; it killed 10 runs of the production campaign (every
# chain-3 MedPois run) at sweep 1. The exact call is kept -- it is the accurate one and the
# fallback is never reached in the normal regime -- with a guard that uses the cdf value we
# already have. log(F) deviates from logcdf by ~3e-8, and when F itself has underflowed the
# largest term of the sum is the right order of magnitude.
@inline function _robust_logcdf(dist, x, F)
    try
        return logcdf(dist, x)
    catch
        return F > 0.0 ? log(F) : logpdf(dist, x)
    end
end
@inline function _robust_logccdf(dist, x, F)
    try
        return logccdf(dist, x)
    catch
        return F < 1.0 ? log1p(-F) : logpdf(dist, x + 1)
    end
end

@inline function os_consts(dist, Y)
    pdfY = pdf(dist, Y)
    FYm1 = cdf(dist, Y - 1)
    # F(Y) = F(Y-1) + f(Y) exactly, so one incomplete gamma / incomplete beta is enough
    # for both. Measured deviation from calling cdf(dist, Y) directly: 9.9e-16, i.e.
    # rounding. It also guarantees FY - pdfY == FYm1 to rounding, which is what the grid
    # pmf wants. NOTE: log(FYm1) is NOT substituted for logcdf(dist, Y-1) -- that one
    # does cost accuracy (2.8e-08) and is kept as a real call, as is logccdf.
    FY = FYm1 + pdfY
    FY > 1.0 && (FY = 1.0)
    OSConsts(pdfY, logpdf(dist, Y), FY, FYm1,
             _robust_logcdf(dist, Y - 1, FYm1), _robust_logccdf(dist, Y, FY))
end

# cdf/ccdf of the j-th order statistic of n iid draws, evaluated through the parent cdf
# value F — identical to Distributions' logcdf(OrderStatistic(dist, n, j), y) which is
# logcdf(Beta{Int}(j, n - j + 1), cdf(dist, y)).
@inline _os_unif(n::Int, j::Int) = Beta{Int}(j, n - j + 1)

# --- shared order-statistic pmf/cdf from PRECOMPUTED parent quantities ---------
# The parent cdf F(y) and pmf f(y) do not depend on the order D or the rank j, so a
# caller sweeping over candidate D (a D-update) or over classification steps should
# evaluate them ONCE and reuse them here. Each of these is bitwise identical to the
# corresponding Distributions call on OrderStatistic(dist, D, j) — verified with
# `===` — but skips re-evaluating the parent's incomplete gamma / incomplete beta.
#
#   logpmf_orderstat(FY, pdfY, D, j) == logpdf(OrderStatistic(dist, D, j), y)
#   logcdf_orderstat(FY, D, j)       == logcdf(OrderStatistic(dist, D, j), y)
#   logccdf_orderstat(FY, D, j)      == logccdf(OrderStatistic(dist, D, j), y)
# where FY = cdf(dist, y) and pdfY = pdf(dist, y).
@inline logpmf_orderstat(FY::Real, pdfY::Real, D::Int, j::Int) =
    Distributions.logdiffcdf(_os_unif(D, j), FY, FY - pdfY)
@inline logcdf_orderstat(FY::Real, D::Int, j::Int) = logcdf(_os_unif(D, j), FY)
@inline logccdf_orderstat(FY::Real, D::Int, j::Int) = logccdf(_os_unif(D, j), FY)

# --- cancellation-free order-statistic log-pmf (restricted-multinomial expansion) ----
# logpmf_orderstat above evaluates P(X_(j)=y) as a DIFFERENCE of two incomplete-beta
# tails, which (a) costs two incomplete-beta evaluations per candidate D and (b) loses
# precision, or underflows to -Inf, when the two tails are close.
#
# Equivalently, classifying the D parent draws as (<y, =y, >y) with counts (a,e,b):
#
#   P(X_(j) = y) = sum_{a=0}^{j-1} sum_{b=0}^{D-j} D!/(a! e! b!) F(y-1)^a f(y)^e S(y)^b,
#                  e = D-a-b >= 1
#
# Every term is POSITIVE, so there is no cancellation, and the sum stays finite in
# regimes where the tail-difference form underflows. Cost is j(D-j+1) fused
# multiply-adds against two incomplete betas -- measured 4.2x faster at Dmax=15 with a
# maximum relative deviation of 3e-13 from the Distributions result.
#
# Build the coefficient table ONCE per (D, j) outside any per-cell loop.
function orderstat_grid_coefs(D::Int, j::Int)
    T = zeros(Float64, j, D - j + 1)
    @inbounds for a in 0:(j-1), b in 0:(D-j)
        e = D - a - b
        T[a+1, b+1] = e >= 1 ?
            exp(SpecialFunctions.loggamma(D + 1) - SpecialFunctions.loggamma(a + 1) -
                SpecialFunctions.loggamma(e + 1) - SpecialFunctions.loggamma(b + 1)) : 0.0
    end
    return T
end

# F1 = cdf(parent, y), f = pdf(parent, y). pA/pE/pB are scratch buffers of length >= D+1
# (pass per-thread buffers). Returns NaN if the sum underflows, so the caller can fall
# back to the incomplete-beta / BigFloat path.
# `logf` is the parent LOG-pmf at y. It defaults to log(f), which reproduces the previous
# behaviour exactly, but callers that have logpdf on hand should pass it: when y sits far
# into the parent's tail, f underflows to exactly 0.0 in Float64 while logpdf stays finite
# (e.g. NegBin(15000, 0.35) at y = 136: pdf 0.0, logpdf -15033.06). Handed f = 0 the
# log-space retry below cannot recover -- the information is already gone from its input --
# so it returns NaN and the caller falls through to the BigFloat path, which recurses at
# escalating precision chasing a value near log(0). That cost 48 GB per sweep and OOM-killed
# a cluster job. With logf supplied the retry returns the correct value (here log(D)+logf,
# the one-draw-at-y term) in Float64.
function logpmf_orderstat_grid(F1::Float64, f::Float64, D::Int, j::Int, T::Matrix{Float64},
                               pA::Vector{Float64}, pE::Vector{Float64}, pB::Vector{Float64},
                               logf::Float64 = (f > 0.0 ? log(f) : -Inf))
    F0 = F1 - f
    F0 < 0.0 && (F0 = 0.0)
    S1 = 1.0 - F1
    S1 < 0.0 && (S1 = 0.0)
    pA[1] = 1.0; pE[1] = 1.0; pB[1] = 1.0
    @inbounds for k in 1:D
        pA[k+1] = pA[k] * F0
        pE[k+1] = pE[k] * f
        pB[k+1] = pB[k] * S1
    end
    s = 0.0
    @inbounds for b in 0:(D-j)
        pb = pB[b+1]
        pb == 0.0 && continue
        for a in 0:(j-1)
            e = D - a - b
            e < 1 && continue
            s += T[a+1, b+1] * pA[a+1] * pE[e+1] * pb
        end
    end
    s > 0.0 && return log(s)

    # Linear-space underflow (e.g. f(y)^e below realmin when y is far above mu, where
    # F(y) has also saturated to 1.0). Retry the SAME positive sum in log space; the
    # coefficients D!/(a!e!b!) are always representable, so log(T) is safe. Without
    # this the caller falls back to logpdf(OrderStatistic(...)), which in this regime
    # returns a finite but badly wrong value -- and being finite it never triggers the
    # BigFloat guard either (e.g. mu=0.3, Y=30, D=15, j=8: truth -879.8, that path -145.2).
    lF0 = F0 > 0.0 ? log(F0) : -Inf
    lf  = logf
    lS1 = S1 > 0.0 ? log(S1) : -Inf
    acc = -Inf
    @inbounds for b in 0:(D-j)
        (b > 0 && isinf(lS1)) && continue
        for a in 0:(j-1)
            e = D - a - b
            e < 1 && continue
            (a > 0 && isinf(lF0)) && continue
            c = T[a+1, b+1]
            c <= 0.0 && continue
            t = log(c) + (a == 0 ? 0.0 : a*lF0) + e*lf + (b == 0 ? 0.0 : b*lS1)
            isfinite(t) || continue
            acc = acc >= t ? acc + log1p(exp(t - acc)) : t + log1p(exp(acc - t))
        end
    end
    return isfinite(acc) ? acc : NaN
end

# --- cancellation-free order-statistic CDFs (finite binomial-tail sums) --------------
# logcdf(_os_unif(n,r), F) = P(Binomial(n,F) >= r) = sum_{k=r}^{n} C(n,k) F^k (1-F)^(n-k)
# logccdf(_os_unif(n,r), F) = P(Binomial(n,F) <= r-1) = sum_{k=0}^{r-1} ...
# Both are finite sums of POSITIVE terms, so they carry no cancellation and cost a
# handful of multiply-adds instead of an incomplete beta. Measured 3.6x faster with a
# maximum relative deviation of 3.65e-15. Return NaN on underflow so callers fall back.
const _OS_NMAX = 64
const _BINOMTAB = Float64[k <= n ? binomial(big(n), big(k)) : 0.0 for n in 0:_OS_NMAX, k in 0:_OS_NMAX]

@inline function logcdf_os_sum(F::Float64, n::Int, r::Int)
    r <= 0 && return 0.0
    r > n && return -Inf
    S = 1.0 - F
    S < 0.0 && (S = 0.0)
    s = 0.0
    @inbounds for k in r:n
        s += _BINOMTAB[n+1, k+1] * F^k * S^(n-k)
    end
    return s > 0.0 ? log(s) : NaN
end

@inline function logccdf_os_sum(F::Float64, n::Int, r::Int)
    r <= 0 && return -Inf
    r > n && return 0.0
    S = 1.0 - F
    S < 0.0 && (S = 0.0)
    s = 0.0
    @inbounds for k in 0:(r-1)
        s += _BINOMTAB[n+1, k+1] * F^k * S^(n-k)
    end
    return s > 0.0 ? log(s) : NaN
end

@inline function _logcdf_os(F::Float64, n::Int, r::Int)
    n > _OS_NMAX && return logcdf(_os_unif(n, r), F)
    v = logcdf_os_sum(F, n, r)
    return isnan(v) ? logcdf(_os_unif(n, r), F) : v
end

@inline function _logccdf_os(F::Float64, n::Int, r::Int)
    n > _OS_NMAX && return logccdf(_os_unif(n, r), F)
    v = logccdf_os_sum(F, n, r)
    return isnan(v) ? logccdf(_os_unif(n, r), F) : v
end

# Cached coefficient tables + scratch buffers, for callers that evaluate the grid pmf
# one (D, j) at a time (e.g. held-out likelihood) rather than sweeping candidate D.
const _GRID_CACHE = Dict{Tuple{Int,Int},Matrix{Float64}}()
const _GRID_LOCK = ReentrantLock()

function orderstat_grid_coefs_cached(D::Int, j::Int)
    lock(_GRID_LOCK) do
        get!(_GRID_CACHE, (D, j)) do
            orderstat_grid_coefs(D, j)
        end
    end
end

# log P(X_(j) = y) for a discrete parent, from the parent's cdf and pmf at y.
# Cancellation-free; returns NaN on underflow so callers can fall back.
function logpmf_orderstat_grid_cached(F1::Float64, f::Float64, D::Int, j::Int,
                                      logf::Float64 = (f > 0.0 ? log(f) : -Inf))
    T = orderstat_grid_coefs_cached(D, j)
    pA = Vector{Float64}(undef, D + 1)
    pE = Vector{Float64}(undef, D + 1)
    pB = Vector{Float64}(undef, D + 1)
    return logpmf_orderstat_grid(F1, f, D, j, T, pA, pE, pB, logf)
end

function sampleSumGivenOrderStatistic(Y,D,j,dist)
    if D == 1
        return Y
    end
    if Y == 0
        if D == j 
            return 0
        end
        # else
        #     start = j 
        # end
    end
    # if pdf(dist,Y) < 1e-100
    #     return Y*D
    # end

    @assert D >= j
    r_lower = 0
    r_highr = 0
    r_equal = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0
    if Y == 0
        if D == j 
            return 0
        end
        # else
        #     start = j 
        # end
    end
    @assert D >= j    
    C = os_consts(dist, Y)
    @views for k in 1:D
        # println(k, " ", j, " ", r_lower, D, " ", r_highr)
        @assert (j - r_lower) >= 1
        @assert (D - r_highr) >= j
        if r_equal == 0 && r_lower + r_highr == D - 1
            r_equal = 1
            break
        end
        if r_equal >= 1 && j - r_lower == 1
            r_higheq = D - k + 1
            break
        elseif r_equal >= 1 && D - r_highr == j
            r_loweq = D - k + 1
            break
        elseif r_equal >= j - r_lower && r_equal >= D - r_highr - j + 1
            r_any = D - k + 1
            break
        end
        l1, l2, l3 = logprobVec2(Y,j,D,dist,r_lower,r_equal,r_highr,C)
        # scalar draws in the same order as rand(Gumbel(0,1),3) -> identical values;
        # argmax on a tuple keeps argmax's first-wins tie-breaking, allocation-free
        g1 = rand(Gumbel(0,1)); g2 = rand(Gumbel(0,1)); g3 = rand(Gumbel(0,1))
        c = argmax((l1 + g1, l2 + g2, l3 + g3))
        if c == 1
            r_lower += 1
        elseif c == 2
            r_equal += 1
        else #c == 3
            r_highr += 1
        end
    end
    @assert r_lower + r_highr + r_equal + r_higheq + r_loweq + r_any == D
    if r_any != 0
        #return Y*r_equal
            return sum(saferand(dist, r_any, Y)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower)) + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    else #at least one of first two will be 0, safeTrunc, if given n=0, returns 0; both can be 0 as well
        #return Y*r_equal
        return sum(safeTrunc(dist, 0, Y,n=r_loweq)) + sum(safeTrunc(dist, Y, Inf,n=r_higheq)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower))  + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    end
    #tasks = Task[]

    # if r_any != 0
    #     push!(tasks, @spawn sum(saferand(dist, r_any, Y)))
    # else
    #     push!(tasks, @spawn sum(safeTrunc(dist, 0, Y, n=r_loweq)))
    #     push!(tasks, @spawn sum(safeTrunc(dist, Y, Inf, n=r_higheq)))
    # end

    # push!(tasks, @spawn sum(safeTrunc(dist, 0, Y-1, n=r_lower)))
    # push!(tasks, @spawn sum(safeTrunc(dist, Y+1, Inf, n=r_highr)))

    # # cheap scalar term stays serial
    # total = Y * r_equal

    # # collect parallel results
    # total += sum(fetch.(tasks))

    # return total
end

# Draw n iid values from `dist` truncated to [lower, upper], deciding feasibility
# by the interval's MASS:
#  - mass > 0.5: rejection from the parent (expected < 2 draws, exact)
#  - intermediate mass: cdf inversion via quantile, clamped to the interval
#    (clamping only binds when float precision is exhausted, where the mass
#    concentrates on that boundary anyway)
#  - numerically zero mass: the boundary atom dominates
function randTruncDirect(dist, lower, upper, n)
    T = eltype(dist)
    n == 0 && return T[]
    out = Vector{T}(undef, n)
    Fl = lower <= 0 ? 0.0 : cdf(dist, lower - 1)
    Fu = isinf(upper) ? 1.0 : cdf(dist, upper)
    mass = Fu - Fl
    if mass > 0.5
        # rejection from the parent: exact, fewer than 2 draws expected
        @inbounds for i in 1:n
            x = rand(dist)
            while x < lower || x > upper
                x = rand(dist)
            end
            out[i] = x
        end
    elseif mass > 0
        # cdf inversion; cap u strictly below 1 so quantile stays finite for
        # integer-valued parents, then clamp (binds only once float precision
        # is exhausted, where the mass sits on that boundary anyway)
        @inbounds for i in 1:n
            u = min(Fl + rand() * mass, prevfloat(1.0))
            x = quantile(dist, u)
            out[i] = isinf(upper) ? max(x, lower) : clamp(x, lower, upper)
        end
    else # numerically zero mass: boundary atom dominates
        fill!(out, isinf(upper) ? lower : upper)
    end
    return out
end

function logprobY2(Y,D,j,dist,numY)
    if numY < j && numY < D - j + 1
        # println("D: ", D, " j: ", j, " Y: ", numY)
        # println("general")
        #println(numY*logpdf(dist,Y) + logsubexp(logcdf(OrderStatistic(dist, D-numY, j-numY), Y), logcdf(OrderStatistic(dist, D-numY, j), Y-1)))
        return numY*logpdf(dist,Y) + logsubexp(logcdf(OrderStatistic(dist, D-numY, j-numY), Y), logcdf(OrderStatistic(dist, D-numY, j), Y-1))
    elseif numY < D - j + 1 && numY >= j
        # println("D: ", D, " j: ", j, " Y: ", numY)
        # println("low")
        return numY*logpdf(dist,Y) + logccdf(OrderStatistic(dist, D-numY, j), Y-1)
    elseif numY < j && numY >= D - j + 1
        # println("D: ", D, " j: ", j, " Y: ", numY)
        # println("high")
        return numY*logpdf(dist,Y) + logcdf(OrderStatistic(dist, D-numY, j-numY), Y)
    elseif numY >= j && numY >= D - j + 1
        #println("D: ", D, " j: ", j, " Y: ", numY)
        return numY*logpdf(dist,Y)
    end
end

# methods taking precomputed OSConsts: same formulas, parent cdf/pdf evaluated once by the caller
function logprobY2(Y,D,j,dist,numY,C::OSConsts)
    if numY < j && numY < D - j + 1
        return numY*C.lpY + logsubexp(_logcdf_os(C.FY, D-numY, j-numY), _logcdf_os(C.FYm1, D-numY, j))
    elseif numY < D - j + 1 && numY >= j
        return numY*C.lpY + _logccdf_os(C.FYm1, D-numY, j)
    elseif numY < j && numY >= D - j + 1
        return numY*C.lpY + _logcdf_os(C.FY, D-numY, j-numY)
    elseif numY >= j && numY >= D - j + 1
        return numY*C.lpY
    end
end

# Returns a TUPLE (not a 3-element Vector): identical values, no heap allocation.
# The two `sum(isinf.(...))` broadcasts of the previous version are also replaced by
# scalar counts (they allocated a BitVector each and were evaluated twice per call).
function logprobVec2(Y,j,D,dist,numUnder,numY,numOver,C::OSConsts)
    if C.pdfY < 1e-300
        v = lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
        return (v[1], v[2], v[3])
    end
    conditionD = D - numUnder - numOver
    conditionj = j - numUnder
    if conditionD == 1
        return (-Inf, 0.0, -Inf)
    end
    if numUnder < j - 1 && Y > 0
        jointYless = logprobY2(Y,conditionD-1,conditionj-1,dist,numY,C)
    else
        jointYless = -Inf
    end
    if numOver < D - j
        jointYmore = logprobY2(Y,conditionD-1,conditionj,dist,numY,C)
    else
        jointYmore = -Inf
    end
    logprobequal = logprobY2(Y,conditionD,conditionj,dist,numY+1,C)
    logprobless = C.lcdfYm1 + jointYless
    logprobmore = C.lccdfY + jointYmore

    if isinf(logprobless) && isinf(logprobequal) && isinf(logprobmore)
        v = lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
        logprobless, logprobequal, logprobmore = v[1], v[2], v[3]
    end
    @assert !(isinf(logprobless) && isinf(logprobequal) && isinf(logprobmore))
    return (logprobless, logprobequal, logprobmore)
end

function logprobVec2(Y,j,D,dist,numUnder,numY,numOver)
    if pdf(dist,Y) < 1e-300
        return lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    end
    conditionD = D - numUnder - numOver
    conditionj = j - numUnder
    if conditionD == 1
        return [-Inf,0,-Inf]
    end
    if numUnder < j - 1 && Y > 0
        jointYless = logprobY2(Y,conditionD-1,conditionj-1,dist,numY)
    else
        jointYless = -Inf
    end
    if numOver < D - j 
        jointYmore = logprobY2(Y,conditionD-1,conditionj,dist,numY)
    else
        jointYmore = -Inf
    end
    logprobequal = logprobY2(Y,conditionD,conditionj,dist,numY+1)
    logprobless = logcdf(dist,Y-1) + jointYless#/jointYdenom
    logprobmore = logccdf(dist,Y) + jointYmore#/jointYdenom

    
    
    logprobs = [logprobless,logprobequal,logprobmore]
    if sum(isinf.(logprobs)) ==  3

        logprobs = lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    end
    @assert sum(isinf.(logprobs)) <  3
    #for checking
    #logdenom = logprobY2(Y,conditionD,conditionj,dist,numY)
    # logprobs = [exp(logprobless),exp(logprobequal),exp(logprobmore)]
    # logprobs = logprobs ./ sum(logprobs)

    

    return logprobs
end

function lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    D = D - numY - numUnder - numOver
    j = j - numUnder
    #println("ahh")
    if numY == 0
        #println("nope")
        if Y > mean(dist)
            probUnder = (j-1)/D
            return [log(probUnder), log(1-probUnder),-Inf]
        else
            probOver = (D-j+numY)/D
            return [-Inf, log(1-probOver),log(probOver)]
        end
    else
        #println("yep")
        if Y > mean(dist) 
            logtruncProb = 0
            if pdf(dist,Y) != 0 
                try
                    logtruncProb = logpdf(Truncated(dist, Y, Inf), Y)
                catch ex
                    logtruncProb = 0
                end
            else
                logtruncProb = 0
            end
            if isnan(logtruncProb) || logtruncProb > 0 
                logtruncProb = 0
            end

            probUnder = (j-1)/D
            return [log(probUnder), log(1-probUnder) + logtruncProb,log(1-probUnder) + log1mexp(logtruncProb)]
        else #Y <= mean(dist)
            logtruncProb = 0
            if pdf(dist,Y) != 0 
                try 
                    logtruncProb = logpdf(Truncated(dist, 0, Y), Y)
                catch ex
                    logtruncProb = 0
                end
            else
                logtruncProb = 0
            end
            if isnan(logtruncProb) || logtruncProb > 0 
                logtruncProb = 0
            end
            probOver = (D-j+numY)/D
            return [log(1-probOver)+ log1mexp(logtruncProb), log(1-probOver) + logtruncProb,log(probOver)]
        end
    end
end

function sampleAllGivenOrderStatistic(Y,D,j,dist)
    if D == 1
        return [Y] 
    end
    if Y == 0
        if D == j 
            return zeros(D)
        end
        # else
        #     start = j 
        # end
    end
    # if pdf(dist,Y) < 1e-100
    #     return Y*D
    # end

    @assert D >= j
    r_lower = 0
    r_highr = 0
    r_equal = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0
    if Y == 0
        if D == j 
            return 0
        end
        # else
        #     start = j 
        # end
    end
    @assert D >= j    
    C = os_consts(dist, Y)
    @views for k in 1:D
        # println(k, " ", j, " ", r_lower, D, " ", r_highr)
        @assert (j - r_lower) >= 1
        @assert (D - r_highr) >= j
        if r_equal == 0 && r_lower + r_highr == D - 1
            r_equal = 1
            break
        end
        if r_equal >= 1 && j - r_lower == 1
            r_higheq = D - k + 1
            break
        elseif r_equal >= 1 && D - r_highr == j
            r_loweq = D - k + 1
            break
        elseif r_equal >= j - r_lower && r_equal >= D - r_highr - j + 1
            r_any = D - k + 1
            break
        end
        l1, l2, l3 = logprobVec2(Y,j,D,dist,r_lower,r_equal,r_highr,C)
        # scalar draws in the same order as rand(Gumbel(0,1),3) -> identical values;
        # argmax on a tuple keeps argmax's first-wins tie-breaking, allocation-free
        g1 = rand(Gumbel(0,1)); g2 = rand(Gumbel(0,1)); g3 = rand(Gumbel(0,1))
        c = argmax((l1 + g1, l2 + g2, l3 + g3))
        if c == 1
            r_lower += 1
        elseif c == 2
            r_equal += 1
        else #c == 3
            r_highr += 1
        end
    end
    @assert r_lower + r_highr + r_equal + r_higheq + r_loweq + r_any == D

    out = Int64[]   # or Vector{eltype(dist)}()

    if r_any != 0
        append!(out, saferand(dist, r_any, Y))
    else
        r_loweq  != 0 && append!(out, safeTrunc(dist, 0, Y;   n = r_loweq))
        r_higheq != 0 && append!(out, safeTrunc(dist, Y, Inf; n = r_higheq))
    end

    r_equal != 0 && append!(out, fill(Y, r_equal))
    r_lower != 0 && append!(out, safeTrunc(dist, 0, Y-1; n = r_lower))
    r_highr != 0 && append!(out, safeTrunc(dist, Y+1, Inf; n = r_highr))

    return out
end