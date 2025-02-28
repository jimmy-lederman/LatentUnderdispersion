using Distributions
using LogExpFunctions
# include("PoissonMedianFunctions.jl")

function safeTrunc(dist, lower, upper; n=1)
    if n == 0
        return 0
    end
    #try
    return rand(Truncated(dist, lower, upper), n)
    # catch e
    #     if lower == 0
    #         return fill(upper, n)
    #     elseif isinf(upper)
    #         return fill(lower, n)
    #     end
    # end
end

function logprobYatIterationMax(Y,i,dist)
    num = logpdf(dist,Y) + (i-1)*logcdf(dist, Y)
    denom = logsubexp(i*logcdf(dist, Y), i*logcdf(dist, Y-1))
    return num - denom
end

function logprobYatIterationMin(Y,i,dist)
    num = logpdf(dist,Y) + (i-1)*logccdf(dist, Y-1)
    denom = logsubexp(i*logccdf(dist, Y-1), i*logccdf(dist, Y))
    return num - denom
end

function log1mexp(a)
    if a < log(2)
        return log(-1*expm1(a))
    else
        return log1p(-1*exp(a))
    end
end

function sampleIndex(Y,D,dist,type=1)
    #this is an approximation for stability; take out if
    #testing for correctness
    if pdf(dist, Y) < 1e-100
        m = mean(dist)
        if (Y > m && type == 1) || (Y < m && type == 2)
            return rand(DiscreteUniform(1,D))
        else
            return 1
        end
    end
    index = 1
    logprevprobs = 0
    totalmasstried = 0
    @views for d in D:-1:1
        if type == 1 #max
            logprobY = logprobYatIterationMax(Y,d,dist)
        else #type == 2, min
            logprobY = logprobYatIterationMin(Y,d,dist)
        end
        if d == D
            logstopprobtemp = logprobY
        else
            logstopprobtemp = logprobY + logprevprobs
        end
        if totalmasstried == 0
            logstopprob = logstopprobtemp
        else
            logstopprob = logstopprobtemp - log1mexp(totalmasstried)
        end
        stopprob = exp(logstopprob)
        if abs(stopprob - 1) < 10e-5
            break
        end
        
        stop = rand(Bernoulli(stopprob))

        if stop
            break
        end
        if d == 2
            index = D
            break
        end
        if totalmasstried == 0
            totalmasstried = logstopprobtemp
        else
            totalmasstried = logsumexp(totalmasstried, logstopprobtemp)
        end
        logprevprobs += log1mexp(logprobY)
        index += 1
    end

    return index
end

function sampleSumGivenMax(Y,D,dist)
    if Y == 0
        return 0
    end
    index = sampleIndex(Y,D,dist,1)
    sample1 = safeTrunc(dist, 0, Y - 1, n=index - 1)
    sample2 = safeTrunc(dist, 0, Y, n=D - index)
    return sum(sample1) + Y + sum(sample2)
end

function sampleSumGivenMin(Y,D,dist)
    index = sampleIndex(Y,D,dist,2)
    sample1 = safeTrunc(dist, Y + 1, Inf, n=index - 1)
    sample2 = safeTrunc(dist, Y, Inf, n=D - index)
    return sum(sample1) + Y + sum(sample2)
end


#####
#code for median of 3 sampler
#########
function numericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    D = D - numY - numUnder - numOver
    j = j - numUnder
    #println("ahh")
    if numY == 0
        #println("nope")
        if Y > mean(dist)
            probUnder = (j-1)/D
            return [probUnder, (1-probUnder),0]
        else
            probOver = (D-j+numY)/D
            return [0, (1-probOver),probOver]
        end
    else
        #println("yep")
        if Y > mean(dist) 
            truncProb = 1
            try 
                truncProb = pdf(Truncated(dist, Y, Inf), Y)
            catch ex
                truncProb = 1
            end
            if isnan(truncProb) || isinf(truncProb)
                truncProb = 1
            end
            probUnder = (j-1)/D
            return [probUnder, (1-probUnder)*truncProb,(1-probUnder)*(1-truncProb)]
        else #Y <= mean(dist)
            truncProb = 1
            try 
                truncProb = pdf(Truncated(dist, 0, Y), Y)
            catch ex
                truncProb = 1
            end
            if isnan(truncProb) || isinf(truncProb)
                truncProb = 1
            end
            probOver = (D-j+numY)/D
            return [(1-probOver)*(1-truncProb), (1-probOver)*truncProb,probOver]
        end
    end
end

function logcategorical1(y,dist)
    if pdf(dist,y) < 1e-150
        probs = numericalProbs(y,2,3,dist,0,0,0)
        for i in 1:3
            if probs[i] < 1e-8
                probs[i] = 0
            end
        end
        return rand(Categorical(probs/sum(probs)))
    else
        prob1 = logcdf(dist, y-1) + logpdf(OrderStatistic(dist, 2, 1), y)
        prob2 = logsumexp(log(2) + logcdf(dist,y-1) + logccdf(dist,y), logpdf(dist,y) + logsubexp(log(2),logpdf(dist,y))) + logpdf(dist,y)
        prob3 = logccdf(dist, y) + logpdf(OrderStatistic(dist, 2, 2), y)
        probs = [prob1,prob2,prob3]
        #println(exp.(probs)./sum(exp.(probs)))
        return argmax(probs .+ rand(Gumbel(0,1),3))
    end
end

function logcategorical2(y,dist)
    if pdf(dist,y) < 1e-150
        probs = numericalProbs(y,2,3,dist,0,1,0)
        for i in 1:3
            if probs[i] < 1e-8
                probs[i] = 0
            end
        end
        return rand(Categorical(probs/sum(probs)))
    else
        prob1 = logcdf(dist, y-1) + logccdf(dist, y-1)
        prob2 = logpdf(dist,y)
        prob3 = logccdf(dist, y) + logcdf(dist,y)  
        probs = [prob1,prob2,prob3]
        #println(exp.(probs)./sum(exp.(probs)))
        return argmax(probs .+ rand(Gumbel(0,1),3))
    end
end

#I should break down probability of each event
#to try to see a pattern
function sampleSumGivenMedian3(Y,dist)
    #draw c
    #do a numeric test
    c = logcategorical1(Y,dist)
    if c == 1 #Z1 < Y
        result = safeTrunc(dist, 0, Y - 1)[1] + sampleSumGivenMin(Y,2,dist)
    elseif c == 3 #Z1 > Y 
        result = safeTrunc(dist, Y+1, Inf)[1] + sampleSumGivenMax(Y,2,dist)
    else #Z1 == Y
        #draw new c 
        #do a numeric test
        c = logcategorical2(Y,dist)
        if c == 1
            result = Y + safeTrunc(dist, 0, Y - 1)[1] + safeTrunc(dist, Y, Inf)[1]
        elseif c == 3
            result = Y + safeTrunc(dist, Y+1, Inf)[1] + safeTrunc(dist, 0, Y)[1]
        else #Z1 and Z2 = Y
            result = 2*Y + rand(dist)
        end
    end
    return result
end

function sampleSumGivenOrderStatistic(Y,D,j,dist)
    @assert D >= j
    #special edge cases for efficiency
    if D == 1
        return Y
    elseif j == 1
        return sampleSumGivenMin(Y,D,dist)
    elseif j == D 
        return sampleSumGivenMax(Y,D,dist)
    # elseif D == 3 && j == 2
    #     return sampleSumGivenMedian3(Y,dist)
    else
        #@assert 1 == 2
        return sampleSumGivenOrderStatisticAll(Y,D,j,dist)
    end
end

function sampleSumGivenOrderStatistic2(Y,D,j,dist)
    @assert D >= j
    r_lower = 0
    r_highr = 0
    r_equal = 0
    r_higheq = 0
    r_loweq = 0
    r_any = 0
    @views for k in 1:D
        # @assert j - r_lower >= 1
        # @assert D - r_highr >= j
        # if r_equal >= 1 && j - r_lower == 1
        #     r_higheq = D - k + 1
        #     break
        # elseif r_equal >= 1 && D - r_highr == j
        #     r_loweq = D - k + 1
        #     break
        # elseif r_equal >= j - r_lower && r_equal >= D - r_highr + 1
        #     r_any = D - k + 1
        #     println("any")
        #     @assert 1 == 2
        #     break
        # end
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
        return sum(rand(dist, r_any)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower)) + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    else #at least one of first two will be 0, safeTrunc, if given n=0, returns 0; both can be 0 as well
        return sum(safeTrunc(dist, 0, Y,n=r_loweq)) + sum(safeTrunc(dist, Y, Inf,n=r_higheq)) + Y*r_equal + sum(safeTrunc(dist, 0, Y-1,n=r_lower)) + sum(safeTrunc(dist, Y + 1, Inf,n=r_highr))
    end
end


function sampleSumGivenOrderStatisticAll(Y,D,j,dist)
    numY = 0
    numUnder = 0
    numOver = 0
    total = 0
    effectiveD = 0
    effectivej = 0
    for i in 1:D
        if i > 1
            # if numY == 0 && effectiveD == effectivej
            #     return total + sampleSumGivenMax(Y,effectiveD,dist)
            # elseif numY == 0 && effectivej == 1
            #     return total + sampleSumGivenMin(Y,effectiveD,dist)
            # #end
            # elseif numY == max(j,D-j+1)
            #     return total + sum(rand(dist, D-i+1))
            # end
        end
        logprobs = logprobVec(Y,j,D,dist,numUnder,numY,numOver)
        #println(probs)
        #c = rand(Categorical(probs))
        c = argmax(rand(Gumbel(0,1),3) .+ logprobs)
        if c == 1
            numUnder += 1
            total += safeTrunc(dist, 0, Y-1)[1]
        elseif c == 2
            numY += 1
            total += Y
        else #c == 3
            numOver += 1
            total += safeTrunc(dist, Y + 1, Inf)[1]
        end
        effectiveD = D - numUnder - numOver - numY
        effectivej = j - numUnder
    end
    return total
end

function probY(Y,D,j,dist,numY)
    # println(D, " ", j, " ", numY)
    if numY >= max(j,D-j+1) #constraints filled
        return pdf(dist,Y)^numY
    elseif numY >= j 
        return pdf(dist,Y)^numY * cdf(OrderStatistic(dist, D-numY, j), Y-1)
    elseif D-numY <= j 
        return pdf(dist,Y)^numY * cdf(OrderStatistic(dist, D-numY, j-numY), Y)
    elseif D == j #maximum
        return pdf(dist, Y)^(numY)*cdf(dist,Y)^(D-numY)
    elseif D -1 == j #new variable hits a max
        return pdf(dist, Y)^(numY-1)*(pdf(dist,Y)*cdf(OrderStatistic(dist, D-numY, j-numY), Y) - ccdf(dist,Y-1)*cdf(OrderStatistic(dist, D-numY, j), Y))
    elseif j == 1 #minimum
        return pdf(dist, Y)^(numY)*ccdf(dist,Y-1)^(D-numY)
    elseif j == 2 #new variable hits a minimum
        return pdf(dist, Y)^(numY-1)*(cdf(dist,Y)*cdf(OrderStatistic(dist, D-numY, j-numY), Y) - pdf(dist,Y)*cdf(OrderStatistic(dist, D-numY, j), Y))
    else #all else
        return pdf(dist,Y)^numY * (cdf(OrderStatistic(dist, D-numY, j-numY), Y) - cdf(OrderStatistic(dist, D-numY, j), Y-1))
    end
end

function logprobY(Y,D,j,dist,numY)
    try
        if numY >= max(j,D-j+1) #constraints filled
            return numY*logpdf(dist,Y)
        elseif numY >= j 
            return numY*logpdf(dist,Y) + logcdf(OrderStatistic(dist, D-numY, j), Y-1)
        elseif D-numY <= j 
            return numY*logpdf(dist,Y) + logcdf(OrderStatistic(dist, D-numY, j-numY), Y)
        elseif D == j #maximum
            return numY*logpdf(dist,Y) + (D-numY)*logcdf(dist,Y)
        elseif D -1 == j #new variable hits a max
            return (numY-1)*logpdf(dist,Y) + logsubexp(logpdf(dist,Y)+ logcdf(OrderStatistic(dist, D-numY, j-numY), Y), logccdf(dist,Y-1) + logcdf(OrderStatistic(dist, D-numY, j), Y))
        elseif j == 1 #minimum
            return numY*logpdf(dist,Y) + (D-numY)*logccdf(dist,Y-1)
        elseif j == 2 #new variable hits a minimum
            return (numY-1)*logpdf(dist,Y) + logsubexp(logcdf(dist,Y) + logcdf(OrderStatistic(dist, D-numY, j-numY), Y), logpdf(dist,Y) + logcdf(OrderStatistic(dist, D-numY, j), Y))
        else #all else
            return numY*logpdf(dist,Y) + logsubexp(logcdf(OrderStatistic(dist, D-numY, j-numY), Y), logcdf(OrderStatistic(dist, D-numY, j), Y-1))
        end
    catch ex
        println(D, " ", j, " ", numY)
        @assert 1 == 2
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
    if pdf(dist,Y) < 1e-50
        return lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    end
    conditionD = D - numUnder - numOver
    conditionj = j - numUnder
    if conditionD == 1
        return [-Inf,0,-Inf]
    end
    if numUnder < j - 1
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
    # println(logprobY2(Y,1,1,dist,1))
    # @assert 1 == 2
    #println(logprobs)
    return logprobs
end


function logprobVec(Y,j,D,dist,numUnder,numY,numOver)
    if pdf(dist,Y) < 1e-50
        return lognumericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    end
    conditionD = D - numUnder - numOver
    conditionj = j - numUnder
    if conditionD == 1
        return [-Inf,0,-Inf]
    end
    if numUnder < j - 1
        if numY == 0
            #this seems very wrong!
            jointYless = logcdf(OrderStatistic(dist,conditionD-1,conditionj-1), Y)
        else
            jointYless = logprobY(Y,conditionD-1,conditionj-1,dist,numY)
        end
    else
        jointYless = -Inf
    end
    if numOver < D - j 
        if numY == 0
            #this seems very wrong!
            jointYmore = logcdf(OrderStatistic(dist,conditionD-1,conditionj), Y)
        else
            jointYmore = logprobY(Y,conditionD-1,conditionj,dist,numY)
        end
    else
        jointYmore = -Inf
    end
    logprobequal = logprobY(Y,conditionD,conditionj,dist,numY+1)
    logprobless = logcdf(dist,Y-1) + jointYless#/jointYdenom
    logprobmore = logccdf(dist,Y) + jointYmore#/jointYdenom
    logprobs = [logprobless,logprobequal,logprobmore]
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
            try 
                logtruncProb = logpdf(Truncated(dist, Y, Inf), Y)
            catch ex
                logtruncProb = 0
            end
            if isnan(logtruncProb) || logtruncProb > 0 
                logtruncProb = 0
            end

            probUnder = (j-1)/D
            return [log(probUnder), log(1-probUnder) + logtruncProb,log(1-probUnder) + log1mexp(logtruncProb)]
        else #Y <= mean(dist)
            logtruncProb = 0
            try 
                logtruncProb = logpdf(Truncated(dist, 0, Y), Y)
            catch ex
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