include("OrderStatsSampling.jl")

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