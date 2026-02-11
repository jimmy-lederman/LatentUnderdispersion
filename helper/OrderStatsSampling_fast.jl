using Distributions
using LogExpFunctions
using Random
# include("PoissonMedianFunctions.jl")

function rand_trunc_discrete(dist, lower, upper, n)
    n == 0 && return 0

    Fl = lower <= 0 ? 0.0 : cdf(dist, lower - 1)
    Fu = isinf(upper) ? 1.0 : cdf(dist, upper)

    width = Fu - Fl
    width <= 0 && return fill(lower, n)

    u = rand(n) .* width .+ Fl
    return quantile.(Ref(dist), u)
end

@inline function logcdf_orderstat(dist, D, j, Y, FY)
    # P(X_(j) ≤ Y)
    return logccdf(Binomial(D, FY), j-1)
end

@inline function logccdf_orderstat(dist, D, j, Y, FY)
    # P(X_(j) > Y)
    return logcdf(Binomial(D, FY), j-1)
end


function logprobY2_fast(Y, D, j, dist, numY, FY, FYm1, lpY)
    # FY  = cdf(dist,Y)
    # FYm1 = cdf(dist,Y-1)
    # lpY = logpdf(dist,Y)

    if numY < j && numY < D - j + 1
        return numY*lpY +
               logsubexp(
                   logcdf_orderstat(dist, D-numY, j-numY, Y, FY),
                   logcdf_orderstat(dist, D-numY, j, Y-1, FYm1)
               )

    elseif numY < D - j + 1 && numY >= j
        return numY*lpY +
               logccdf_orderstat(dist, D-numY, j, Y-1, FYm1)

    elseif numY < j && numY >= D - j + 1
        return numY*lpY +
               logcdf_orderstat(dist, D-numY, j-numY, Y, FY)

    else
        return numY*lpY
    end
end


function logprobVec2_fast(
    Y, j, D, dist,
    numUnder, numY, numOver,
    FY, FYm1, lpY
)

    conditionD = D - numUnder - numOver
    conditionj = j - numUnder

    conditionD == 1 && return (-Inf, 0.0, -Inf)

    jointYless = -Inf
    if numUnder < j - 1 && Y > 0
        jointYless = logprobY2_fast(
            Y, conditionD-1, conditionj-1,
            dist, numY, FY, FYm1, lpY
        )
    end

    jointYmore = -Inf
    if numOver < D - j
        jointYmore = logprobY2_fast(
            Y, conditionD-1, conditionj,
            dist, numY, FY, FYm1, lpY
        )
    end

    logprobequal =
        logprobY2_fast(
            Y, conditionD, conditionj,
            dist, numY+1, FY, FYm1, lpY
        )

    logprobless = log(FYm1) + jointYless
    logprobmore = log1p(-FY) + jointYmore

    return (logprobless, logprobequal, logprobmore)
end

@inline function gumbel_argmax3(l1, l2, l3)
    g1 = rand(Gumbel())
    g2 = rand(Gumbel())
    g3 = rand(Gumbel())

    a = l1 + g1
    b = l2 + g2
    c = l3 + g3

    if a > b
        return a > c ? 1 : 3
    else
        return b > c ? 2 : 3
    end
end



function sampleSumGivenOrderStatistic_fast(Y, D, j, dist)

    D == 1 && return Y
    Y == 0 && D == j && return 0

    lpY  = logpdf(dist, Y)
    FY   = cdf(dist, Y)
    FYm1 = Y > 0 ? cdf(dist, Y-1) : 0.0

    r_lower = 0
    r_equal = 0
    r_highr = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0

    @inbounds for k in 1:D

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
        elseif r_equal >= j - r_lower &&
               r_equal >= D - r_highr - j + 1
            r_any = D - k + 1
            break
        end

        l1,l2,l3 = logprobVec2_fast(
            Y, j, D, dist,
            r_lower, r_equal, r_highr,
            FY, FYm1, lpY
        )

        c = gumbel_argmax3(l1,l2,l3)

        if c == 1
            r_lower += 1
        elseif c == 2
            r_equal += 1
        else
            r_highr += 1
        end
    end

    total = Y*r_equal

    total += sum(rand_trunc_discrete(dist, 0, Y-1, r_lower))
    total += sum(rand_trunc_discrete(dist, Y+1, Inf, r_highr))

    if r_any != 0
        total += sum(rand(dist, r_any))
    else
        total += sum(rand_trunc_discrete(dist, 0, Y, r_loweq))
        total += sum(rand_trunc_discrete(dist, Y, Inf, r_higheq))
    end

    return total
end
