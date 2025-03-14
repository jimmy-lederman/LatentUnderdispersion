#using Distributions
#using QuadGK
using SpecialFunctions
using LogExpFunctions
using HypergeometricFunctions

# function log_incomplete_beta2(a, b, x; precision=16)
#     setprecision(BigFloat,precision)
#     a = BigFloat(a)
#     b = BigFloat(b)
#     x = BigFloat(x)
#     #result, _ = quadgk(t -> t^(a-1) * (1-t)^(b-1), BigFloat(0), x; rtol=1e-75)
#     if a >= b
#         result, _ = quadgk(t -> exp((a-1)*log(t) + (b-1)*log(1-t)), BigFloat(0), x; rtol=1e-50)
#         return log(result) - log(SpecialFunctions.beta(a,b))
#     else
#         result, _ = quadgk(t -> exp((b-1)*log(t) + (a-1)*log(1-t)), BigFloat(0), 1-x; rtol=1e-50)
#         # println(result)
#         # flush(stdout)
#         return log1p(-1*result/SpecialFunctions.beta(a,b))
#     end
#     return result
# end

function log_incomplete_beta3(a, b, x; precision=64)
    setprecision(BigFloat,precision)
    a = BigFloat(a)
    b = BigFloat(b)
    x = BigFloat(x)
    #println("inner: ", precision, " ", a, " ", b)
    try
        return a*log(x) + log(pFq((a, 1-b,), (a+1,), x)) - log(a) - log(beta(a,b))
    catch ex
        # if precision > 100000
        #     return 0
        # else 
        #     return log_incomplete_beta3(a,b,x;precision=5*precision)
        # end
        return log_incomplete_beta3(a,b,x;precision=5*precision)
    end
end

function logprobmaxnb(Y,r,p,D;precision=1000)
    #println("outer: ", precision, " ", a, " ", b)
    first =D*log_incomplete_beta3(r,Y+1,p,precision=precision)
    second =D*log_incomplete_beta3(r,Y,p,precision=precision)
    result = logsubexp(first, second)
    
    if isinf(result) || isnan(result) || first >= 0 || second >= 0
        # if precision > 100000
        #     return 0
        # else
        #     return logprobmaxnb(Y,r,p,D;precision=precision*5)
        # end
        return logprobmaxnb(Y,r,p,D;precision=precision*5)
    else
        return Float64(result)
    end
    #return 0
end

function logpmfMaxNegBin(Y,r,p,D)
    llik = logpdf(OrderStatistic(NegativeBinomial(r,p), D, D), Y)
    if isinf(llik) || isnan(llik)
        #try
            llik = logprobmaxnb(Y,r,p,D)
            if llik == 0
                # if Y > mean(NegativeBinomial(r,p))
                #     llik = logpdf(NegativeBinomial(r,p), Y)
                #     @assert llik != 0 && !isinf(llik) && !isnan(llik)
                # else 
                    println(Y, " ", r, " ", p)
                    throw("missed low error")
                #end
            end
        # catch InterruptException
        #     println(Y, " ", r, " ", p)
        #     @assert 1 == 2
        # end
    end
    return llik
end

#need to do: implement median

function logprobOrderStatisticNB(Y,r,p,D,j;precision=1000)
    #This is based on the alternative form of (2.1.3) on page 10 of David's Order Statistics
    #must set precision
    setprecision(BigFloat,precision)
    r = big(r)
    p = big(p)
    Y = big(Y)
    firstbeta = log_incomplete_beta3(r,Y+1,p,precision=precision)
    secondbeta = log_incomplete_beta3(r,Y,p,precision=precision)
    #println(firstbeta, " ", secondbeta)
    if firstbeta >= 0 || secondbeta >= 0
        return logprobOrderStatisticNB(Y,r,p,D,j,precision=5*precision)
    end
    firstbeta_comp = log1mexp(firstbeta)
    secondbeta_comp = log1mexp(secondbeta)
    # println(firstbeta)
    # println(secondbeta)
    # println(firstbeta_comp)
    # println(secondbeta_comp)
    cdf1 = 0 #first term is log(choose(j-1,j-1))
    cdf2 = 0
    for i in 1:(D-j)
        part1 = logabsbinomial(j+i-1,j-1)[1] + i*firstbeta_comp
        part2 = logabsbinomial(j+i-1,j-1)[1] + i*secondbeta_comp
        #result = logsumexp(result, logsubexp(part1,part2))
        cdf1 = logsumexp(cdf1, part1)
        cdf2 = logsumexp(cdf2, part2)
    end
    
    result = logsubexp(j*firstbeta + cdf1, j*secondbeta + cdf2)
    if isinf(result) || isnan(result)
        return logprobOrderStatisticNB(Y,r,p,D,j,precision=5*precision)
    else
        return Float64(result)
    end
end

function logpmfOrderStatNegBin(Y,r,p,D,j)
    llik = logpdf(OrderStatistic(NegativeBinomial(r,p), D, j), Y)
    if isinf(llik) || isnan(llik)
        llik = logprobOrderStatisticNB(Y,r,p,D,j)
        if llik == 0
            println(Y, " ", r, " ", p)
            throw("missed low error")
        end
    end
    return llik
end


# function logprobMedianNegBin(Y,mu)
#     #must set precision
#     setprecision(BigFloat,precision)
#     mu = big(mu)
#     Y = big(Y)
#     firstgammas = gamma_inc(Y+1,mu)
#     secondgammas =gamma_inc(Y,mu)
#     result = logsubexp(2*log(firstgammas[2]) + log(1+2*firstgammas[1]), 2*log(secondgammas[2]) + log(1+2*secondgammas[1]))
#     if isinf(result) || isnan(result)
#         return logprobMedian(Y,mu,precision=5*precision)
#     else
#         return Float64(result)
#     end
# end