using Distributions
using LogExpFunctions
using Random
# include("PoissonMedianFunctions.jl")

function safeTrunc(dist, lower, upper; n=1)
    if n == 0
        return 0
    end
    # do not attempt to try-catch instead of using this condition
    #if the truncation point is sufficiently far in the tail of dist,
    #then it will take a long time to attempt to sample from it and fail
    if (lower != 0 && pdf(dist,lower) > 1e-300) || pdf(dist,upper) > 1e-300
        try
            return rand(Truncated(dist, lower, upper), n)
        catch ex
            if lower == 0
                return fill(upper, n)
            elseif isinf(upper)
                return fill(lower, n)
            end
        end
    else
        if lower == 0
            return fill(upper, n)
        elseif isinf(upper)
            return fill(lower, n)
        end
    end
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

# function safeTrunc2(dist, lower, upper; n=1)
#     if n == 0
#         return []
#     end
#     if pdf(dist,lower) != 0 || pdf(dist,upper) != 0
#         return rand(Truncated(dist, lower, upper), n)
#     else
#         if lower == 0
#             return fill(upper, n)
#         elseif isinf(upper)
#             return fill(lower, n)
#         end
#     end
# end

function expnormalizeCat(x)
    cdf = cumsum(exp.(x .- maximum(x)))     # the exp-normalize trick
    z = last(cdf)
    u = rand()  # uniform(0, 1)
    return searchsortedlast(cdf, u * z) + 1
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
    @views for k in 1:D
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
        # if j - r_lower == 1
        #     logprobs = [-Inf, -1, -1]
        # elseif D - r_highr == j
        #     logprobs = [-1, -1, -Inf]
        # else 
        #     logprobs = [-1, -1, -1]
        # end
        logprobs = logprobVec2(Y,j,D,dist,r_lower,r_equal,r_highr)
        c = argmax(rand(Gumbel(0,1),3) .+ logprobs)
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


function sampleListGivenOrderStatistic2(Y,D,j,dist)
    if D == 1
        return Y 
    end
    @assert D >= j
    if Y == 0
        if D == j 
            return 0
        end
    end

    @assert D >= j
    r_lower = 0
    r_highr = 0
    r_equal = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0
    @views for k in 1:D #just sample first one
        #println(r_lower)
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
        logprobs = logprobVec2(Y,j,D,dist,r_lower,r_equal,r_highr)
        #println(logprobs)
        c = argmax(rand(Gumbel(0,1),3) .+ logprobs)
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
        #@assert 1 == 2
        lst = vcat(rand(dist, r_any), fill(Y,r_equal), safeTrunc2(dist, 0, Y-1,n=r_lower), safeTrunc2(dist, Y + 1, Inf,n=r_highr))
        return shuffle(lst)
    else #at least one of first two will be 0, safeTrunc, if given n=0, returns 0; both can be 0 as well
        lst = vcat(safeTrunc2(dist, 0, Y,n=r_loweq), safeTrunc2(dist, Y, Inf,n=r_higheq), fill(Y,r_equal), safeTrunc2(dist, 0, Y-1,n=r_lower), safeTrunc2(dist, Y + 1, Inf,n=r_highr))
        return shuffle(lst)
    end
end

function sampleFirstKGivenOrderStatistic2(Y,D,j,dist,K)
    if D == 1
        return Y 
    end
    @assert D >= j
    if Y == 0
        if D == j 
            return 0
        end
        # else
        #     D = D - j + 1
        #     j = 1
        # end
    end


    @assert D >= j
    r_lower = 0
    r_highr = 0
    r_equal = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0
    @views for k in 1:K 
        #println(r_lower)
        @assert (j - r_lower) >= 1
        @assert (D - r_highr) >= j
        if r_equal == 0 && r_lower + r_highr == D - 1
            r_equal = 1
            break
        end
        if r_equal >= 1 && j - r_lower == 1
            r_higheq = K - k + 1
            break
        elseif r_equal >= 1 && D - r_highr == j
            r_loweq = K - k + 1
            break
        elseif r_equal >= j - r_lower && r_equal >= D - r_highr - j + 1
            r_any = K - k + 1
            break
        end
        logprobs = logprobVec2(Y,j,D,dist,r_lower,r_equal,r_highr)
        #println(logprobs)
        c = argmax(rand(Gumbel(0,1),3) .+ logprobs)
        if c == 1
            r_lower += 1
        elseif c == 2
            r_equal += 1
        else #c == 3
            r_highr += 1
        end
    end
    @assert r_lower + r_highr + r_equal + r_higheq + r_loweq + r_any == K
    if r_any != 0
        #@assert 1 == 2
        return sum(rand(dist, r_any)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower)) + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    else #at least one of first two will be 0, safeTrunc, if given n=0, returns 0; both can be 0 as well
        return sum(safeTrunc(dist, 0, Y,n=r_loweq)) + sum(safeTrunc(dist, Y, Inf,n=r_higheq)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower)) + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    end
end