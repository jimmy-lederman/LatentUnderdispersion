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