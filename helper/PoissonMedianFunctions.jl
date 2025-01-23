include("PoissonMaxFunctions.jl")
include("PoissonMinFunctions.jl")
# rewriting = pyimport("sympy.codegen.rewriting")
# cfunctions = pyimport("sympy.codegen.cfunctions")

using Distributions

function poisson_cdf_precise(Y,mu;precision=64)
    setprecision(BigFloat,precision)
    mu = big(mu)
    Y = big(Y)
    result = gamma_inc(Y+1,mu)[2]
    if result == 0 || result == 1
        return poisson_cdf_precise(Y,mu,precision=5*precision)
    else
        return result
    end
end

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

function logcategorical1(y,dist)
    if pdf(dist,y) < 1e-150
        probs = numericalProbs(y,2,3,dist,0,0,0)
    else
        prob1 = logcdf(dist, y-1) + logpdf(OrderStatistic(dist, 2, 1), y)
        prob2 = logsumexp(log(2) + logcdf(dist,y-1) + logccdf(dist,y), logpdf(dist,y) + logsubexp(log(2),logpdf(dist,y))) + logpdf(dist,y)
        prob3 = logccdf(dist, y) + logpdf(OrderStatistic(dist, 2, 2), y)
        probs = [prob1,prob2,prob3]
    end
    return argmax(probs .+ rand(Gumbel(0,1),3))
end

function categorical2(y,dist)
    if (y > mean(dist) && pdf(dist,y) < 1e-10) || (y < mean(dist) && pdf(dist,y) < 1e-50)
        probs = numericalProbs(y,2,3,dist,0,1,0)
        # println("using numprobs2")
    else
        prob1 = cdf(dist, y-1)*ccdf(dist, y-1)
        prob2 = pdf(dist,y)
        prob3 = ccdf(dist, y)*cdf(dist,y)  
        probs = [prob1,prob2,prob3]
    end
    #println(probs/sum(probs))
    probs = probs/sum(probs)
    for i in 1:3
        if probs[i] < 1e-10
            probs[i] = 0
        end
    end
    return rand(Categorical(probs/sum(probs)))
end

#I should break down probability of each event
#to try to see a pattern

function sampleSumGivenMedian3(Y,dist)
    #draw c
    #do a numeric test
    c = categorical1(Y,dist)
    if c == 1 #Z1 < Y
        result = safeTrunc(dist, 0, Y - 1)[1] + sampleSumGivenMin(Y,2,dist)
    elseif c == 3 #Z1 > Y 
        result = safeTrunc(dist, Y+1, Inf)[1] + sampleSumGivenMax(Y,2,dist)
    else #Z1 == Y
        #draw new c 
        #do a numeric test
        c = categorical2(Y,dist)
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
            truncProb = pdf(Truncated(dist, Y, Inf), Y)
            if isnan(truncProb)
                truncProb = 1
            end
            probUnder = (j-1)/D
            return [probUnder, (1-probUnder)*truncProb,(1-probUnder)*(1-truncProb)]
        else #Y <= mean(dist)
            truncProb = pdf(Truncated(dist, 0, Y), Y)
            if isnan(truncProb)
                truncProb = 1
            end
            probOver = (D-j+numY)/D
            return [(1-probOver)*(1-truncProb), (1-probOver)*truncProb,probOver]
        end
    end
end

function jointY(Y,j,D,dist,numY)
    #println("D: ", D, " j: ", j, " numY: ", numY)
    #@assert D > 1
    @assert D >= 1
    @assert j >= 1
    @assert numY <= D
    if numY == 0
        return pdf(OrderStatistic(dist, D, j), Y)
    end
    if numY == D
        return pdf(dist,Y)^D
    end
    if j == 1 #min case
        #println("min: ", pdf(dist, Y)^(numY)*ccdf(dist,Y-1)^(D-1))
        return pdf(dist, Y)^(numY)*ccdf(dist,Y-1)^(D-numY)
    elseif D == j #max case
        #println("max: ", pdf(dist, Y)^(numY)*cdf(dist,Y)^(D-1))
        return pdf(dist, Y)^(numY)*cdf(dist,Y)^(D-numY)
        
    elseif numY == max(j,D-j+1) #remaining can be any case
        #println("Ytie: ", pdf(dist,Y)^(numY))
        return pdf(dist,Y)^(numY) #not sure of the exponent here, but think numY
    end
    return cdf(dist,Y-1)*jointY(Y,j-1,D-1,dist,numY) + ccdf(dist,Y)*jointY(Y,j,D-1,dist,numY) + jointY(Y,j,D,dist,numY+1)
end

function probVec(Y,j,D,dist,numUnder,numY,numOver)
    if pdf(dist,Y) < 10e-10
        #println("numerical: ", Y, " ", dist)
        return numericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    end
    conditionD = D - numUnder - numOver
    conditionj = j - numUnder
    if conditionD == 1
        return [0,1,0]
    end
    #jointYdenom = jointY(Y,conditionj,conditionD,dist,numY)
    if numUnder < j - 1
        jointYless = jointY(Y,conditionj-1,conditionD-1,dist,numY)
    else
        jointYless = 0
    end
    if numOver < D - j 
        jointYmore = jointY(Y,conditionj,conditionD-1,dist,numY)
    else
        jointYmore = 0
    end
    jointequal = jointY(Y,conditionj,conditionD,dist,numY+1)
    probless = cdf(dist,Y-1)*jointYless#/jointYdenom
    probmore = ccdf(dist,Y)*jointYmore#/jointYdenom
    probequal = jointequal#/jointYdenom
    probs = [probless,probequal,probmore]
    return probs/sum(probs)
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
    end
    @assert 1 == 2
    numY = 0
    numUnder = 0
    numOver = 0
    total = 0
    for i in 1:D 
        probs = probVec(Y,j,D,dist,numUnder,numY,numOver)
        #println(probs)
        c = rand(Categorical(probs))
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
    end
    return total
end

function logprobMedian(Y,mu;precision=64)
    #must set precision
    setprecision(BigFloat,precision)
    mu = big(mu)
    Y = big(Y)
    firstgammas = gamma_inc(Y+1,mu)
    secondgammas =gamma_inc(Y,mu)
    result = logsubexp(2*log(firstgammas[2]) + log(1+2*firstgammas[1]), 2*log(secondgammas[2]) + log(1+2*secondgammas[1]))
    if isinf(result) || isnan(result)
        return logprobMedian(Y,mu,precision=5*precision)
    else
        return Float64(result)
    end
end

function logpmfOrderStatPoisson(Y,mu,D,j)
    try
        llik = logpdf(OrderStatistic(Poisson(mu), D, j), Y)
        # if isinf(llik) || isnan(llik)
        #     llik = logprob(Y,mu,D)
        # end
        if isinf(llik) || isnan(llik)
            @assert D == 3 && j == 2
            llik = logprobMedian(Y,mu)
        end
        return llik
        
    catch ex
        @assert D == 3 && j == 2
        #println("second")
        llik = logprobMedian(Y,mu)
     
        return llik
    end
end