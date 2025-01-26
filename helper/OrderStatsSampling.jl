using Distributions
using LogExpFunctions

function safeTrunc(dist,lower,upper;n=1)
    try
        return rand(Truncated(dist, lower, upper), n)
    catch e
        if lower == 0
            return fill(upper,n)
        elseif isinf(upper)
            return fill(lower,n)
        end
    end
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
    elseif D == 3 && j == 2
        return sampleSumGivenMedian3(Y,dist)
    else
        throw("unimplemented")
    end
end