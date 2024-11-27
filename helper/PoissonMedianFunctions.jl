include("PoissonMaxFunctions.jl")
include("PoissonMinFunctions.jl")
rewriting = pyimport("sympy.codegen.rewriting")
cfunctions = pyimport("sympy.codegen.cfunctions")

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

function numericalProbs(Y,j,D,dist,numUnder,numY,numOver)
    D = D - numY - numUnder - numOver
    j = j - numUnder
    if numY == 0
        if Y > mean(dist)
            probUnder = (j-1)/D
            return [probUnder, (1-probUnder),0]
        else
            probOver = (D-j+numY)/D
            return [0, (1-probOver),probOver]
        end
    else
        if Y > mean(dist) 
            truncProb = pdf(Truncated(dist, Y, Inf), Y)
            if isnan(truncProb)
                truncProb = 1
            end
            probUnder = (j-1)/D
            return [probUnder, (1-probUnder)*truncProb,(1-probUnder)*(1-truncProb)]
        end
        if Y < mean(dist)
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

function logpmfOrderStatPoisson(Y,mu,D,j)
    try
        llik = logpdf(OrderStatistic(Poisson(mu), D, j), Y)
        # if isinf(llik) || isnan(llik)
        #     llik = logprob(Y,mu,D)
        # end
        if isinf(llik) || isnan(llik)
            @assert D == 3 && j == 2
            #println("first")
            #println("Y: ", Y, " mu: ", mu)
            #llik = logprobsymbolicMedian(Y,mu)
            llik = 0
        end
        return llik
    catch ex
        @assert D == 3 && j == 2
        #println("second")
        #println("Y: ", Y, " mu: ", mu)
        #llik = logprobsymbolicMedian(Y,mu)
        llik = 0
        return llik
    end
end

function logprobsymbolicMedian(s,z)
    s1, z1 = sympy.symbols("s1 z1")
    s2, z2 = sympy.symbols("s2 z2") 
    expm1_opt = rewriting.FuncMinusOneOptim(sympy.exp, cfunctions.expm1)
    part1 = 2*sympy.log(sympy.uppergamma(s1, z1) / sympy.gamma(s1)) + rewriting.optimize(sympy.log(1 + 2*sympy.lowergamma(s1, z1) / sympy.gamma(s1)), rewriting.optims_c99)
    part2 = 2*sympy.log(sympy.uppergamma(s2, z2) / sympy.gamma(s2)) + rewriting.optimize(sympy.log(1 + 2*sympy.lowergamma(s2, z2) / sympy.gamma(s2)), rewriting.optims_c99)
    
    if s != 0
        combine = part1 + sympy.log(-1*expm1_opt(sympy.exp(part2 - part1)-1))
        combine_result = combine.subs(s1, s+1)
        combine_result = combine_result.subs(s2, s)
        combine_result = combine_result.subs(z1, Int(round(z)))
        combine_result = combine_result.subs(z2, Int(round(z)))
        combine_result = sympy.N(combine_result, 100000)
        combine_result = convert(Float64, combine_result)
        return combine_result

    else #if data point is 0
        combine = part1
        combine_result = combine.subs(s1, s+1)
        combine_result = combine_result.subs(z1, z)
        combine_result = sympy.N(combine_result, 100000)
        combine_result = convert(Float64, combine_result)
    end
    return combine_result
end