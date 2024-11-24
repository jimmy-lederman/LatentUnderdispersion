include("PoissonMaxFunctions.jl")
include("PoissonMinFunctions.jl")

using Distributions

# function sampleSumGivenMedian(Y,D,dist)
#     if D == 1
#         return Y 
#     end
#     #@assert D % 2 == 1 #D must be odd
#     D2 = div(D,2) + 1
#     D3 = div(D,2)
#     minsum = sampleSumGivenMin(Y,D2,dist) - Y
#     maxsum = sampleSumGivenMax(Y,D2,dist)
#     rest2 = sum(rand(Truncated(dist, 0, Y), D3))
#     rest = sum(rand(Truncated(dist, Y, Inf), D3))
#     return minsum + maxsum
# end

function categorical1(y,dist)
    #denom = pdf(OrderStatistic(dist, 3, 2), y)
    prob1 = cdf(dist, y-1)*pdf(OrderStatistic(dist, 2, 1), y)
    prob2 = ccdf(dist, y)*pdf(OrderStatistic(dist, 2, 2), y)
    prob3 = (2*cdf(dist,y-1)*ccdf(dist,y)+pdf(dist,y)*(2-pdf(dist,y)))*pdf(dist,y)
    probs = [prob1,prob2,prob3]
    #println(probs/sum(probs))
    return rand(Categorical(probs/sum(probs)))
end

function categorical2(y,dist)
    prob1 = cdf(dist, y-1)*ccdf(dist, y-1)
    prob2 = ccdf(dist, y)*cdf(dist,y)
    prob3 = pdf(dist,y)
    probs = [prob1,prob2,prob3]
    return rand(Categorical(probs/sum(probs)))
end

function sampleSumGivenMedian3(Y,dist)
    #draw c
    c = categorical1(Y,dist)
    if c == 1
        result = rand(Truncated(dist, 0, Y - 1)) + sampleSumGivenMin(Y,2,dist)
    elseif c == 2
        result = rand(Truncated(dist, Y+1, Inf)) + sampleSumGivenMax(Y,2,dist)
    else #Z1 = Y
        #draw new c 
        c = categorical2(Y,dist)
        if c == 1
            result = Y + rand(Truncated(dist, 0, Y - 1)) + rand(Truncated(dist, Y, Inf))
        elseif c == 2
            result = Y + rand(Truncated(dist, Y+1, Inf)) + rand(Truncated(dist, 0, Y))
        else #Z1 and Z2 = Y
            result = 2*Y + rand(dist)
        end
    end
    return result
end

# function probVec(Y,D,j,dist)
#     c = pdf(OrderStatistic(dist, D, j), Y)
#     if D == j == 1
#         return [0,1,0]
#     end
#     if j == 1
#         B = 0
#     else
#         B = cdf(dist,Y-1)*pdf(OrderStatistic(dist, D-1, j-1), Y)/c #probability of z1 < y
#     end
#     if D == j
#         A = 0
#     else 
#         A = ccdf(dist,Y)*pdf(OrderStatistic(dist, D-1, j), Y)/c #probability of z1 > y
#     end
#     if isnan(A) || isnan(B)
#         println(c)
#         println(cdf(dist,Y-1)*pdf(OrderStatistic(dist, D-1, j-1), Y))
#         println(ccdf(dist,Y)*pdf(OrderStatistic(dist, D-1, j), Y))
#         println("Y: ", Y, " ", dist)
#         println([B,1-A-B,A])
#         @assert 1 == 2
#     end
#     return [B,1-A-B,A]
# end

# function sampleFirst(Y,D,j,dist)
#     probs = probVec(Y,D,j,dist)
#     c = rand(Categorical(probs))
#     if c == 1 #Z1 < Y
#         return rand(Truncated(dist, 0, Y-1))
#     elseif c == 3 #Z1 > Y
#         return rand(Truncated(dist, Y+1, Inf))
#     else #Z1 = Y
#         return Y
#     end
# end

function jointY(Y,j,D,dist,numY)
    #println("D: ", D, " j: ", j, " numY: ", numY)
    #@assert D > 1
    if numY == 0
        return pdf(OrderStatistic(dist, D, j), Y)
    end
    if numY == D
        return pdf(dist,Y)^D
    end
    if j == 1
        #println("min: ", pdf(dist, Y)^(numY)*ccdf(dist,Y-1)^(D-1))
        return pdf(dist, Y)^(numY)*ccdf(dist,Y-1)^(D-numY)
    elseif D == j
        #println("max: ", pdf(dist, Y)^(numY)*cdf(dist,Y)^(D-1))
        return pdf(dist, Y)^(numY)*cdf(dist,Y)^(D-numY)
        
    elseif numY == max(j,D-j+1)
        #println("Ytie: ", pdf(dist,Y)^(numY))
        return pdf(dist,Y)^(numY) #not sure of the exponent here, but think numY
    end
    return cdf(dist,Y-1)*jointY(Y,j-1,D-1,dist,numY) + ccdf(dist,Y)*jointY(Y,j,D-1,dist,numY) + jointY(Y,j,D,dist,numY+1)
end

function probVec(Y,j,D,dist,numUnder,numY,numOver)
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
    numY = 0
    numUnder = 0
    numOver = 0
    total = 0
    for i in 1:D 
        probs = probVec(Y,j,D,dist,numUnder,numY,numOver)
        c = rand(Categorical(probs))
        if c == 1
            numUnder += 1
            total += rand(Truncated(dist, 0, Y-1))
        elseif c == 2
            numY += 1
            total += Y
        else #c == 3
            numOver += 1
            total += rand(Truncated(dist, Y + 1, Inf))
        end
    end
    return total
end

