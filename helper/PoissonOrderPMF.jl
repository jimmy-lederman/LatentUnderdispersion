using SpecialFunctions
using LogExpFunctions
using Distributions
#code to compute PMF for Poisson Order Statistic logpmf

# function logprobsymbolic(s,z,D)
#     s1, z1 = sympy.symbols("s1 z1")
#     log_gamma_reg_inc1 = D*sympy.log(sympy.uppergamma(s1, z1) / sympy.gamma(s1))
#     s2, z2 = sympy.symbols("s2 z2")
#     log_gamma_reg_inc2 = D*sympy.log(sympy.uppergamma(s2, z2) / sympy.gamma(s2))
#     if s != 0
#         #combine = sympy.log(sympy.exp(log_gamma_reg_inc1) - sympy.exp(log_gamma_reg_inc2))
#         combine = log_gamma_reg_inc1 + sympy.log(1 - sympy.exp(log_gamma_reg_inc2 - log_gamma_reg_inc1))
#         combine_result = combine.subs(s1, s+1)
#         combine_result = combine_result.subs(z1, Int(round(z))) #rounding the mean is a (small) approximation
#         combine_result = combine_result.subs(z2, Int(round(z))) #rounding the mean is a (small) approximation
#         combine_result = combine_result.subs(s2, s)
#         combine_result = sympy.N(combine_result, 64)
#         combine_result = convert(Float64, combine_result)
#     else #if data point is 0
#         combine = log_gamma_reg_inc1
#         combine_result = combine.subs(s1, s+1)
#         combine_result = combine_result.subs(z1, z)
#         combine_result = sympy.N(combine_result, 64)
#         combine_result = convert(Float64, combine_result)
#     end
#     return combine_result
# end

function logpmfMaxPoisson(Y,mu,D)
    llik = logpdf(OrderStatistic(Poisson(mu), D, D), Y)
    if isinf(llik) || isnan(llik)
         llik = logprobMax(Y,mu,D)
    end
    return llik
end

function logprobMax(Y,mu,D;precision=64)
     #must set precision
     setprecision(BigFloat,precision)
     mu = big(mu)
     Y = big(Y)
     firstgammas = gamma_inc(Y+1,mu)
     secondgammas = gamma_inc(Y,mu)
     result = logsubexp(D*log(firstgammas[2]), D*log(secondgammas[2]))
     if isinf(result) || isnan(result)
        return logprobMax(Y,mu,D,precision=5*precision)
    else
        return Float64(result)
    end
end

function logprobMedian3(Y,mu;precision=1000)
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

function logprobOrderStatisticPoisson(Y,mu,D,j;precision=64)
    #This is based on the alternative form of (2.1.3) on page 10 of David's Order Statistics
    #must set precision
    setprecision(BigFloat,precision)
    mu = big(mu)
    Y = big(Y)
    firstgammas = gamma_inc(Y+1,mu)
    secondgammas = gamma_inc(Y,mu)

    cdf1 = 0 #first term is log(choose(j-1,j-1))
    cdf2 = 0
    for i in 1:(D-j)
        part1 = logabsbinomial(j+i-1,j-1)[1] + i*log(firstgammas[1])
        part2 = logabsbinomial(j+i-1,j-1)[1] + i*log(secondgammas[1])
        #result = logsumexp(result, logsubexp(part1,part2))
        cdf1 = logsumexp(cdf1, part1)
        cdf2 = logsumexp(cdf2, part2)
    end
    result = logsubexp(j*log(firstgammas[2]) + cdf1, j*log(secondgammas[2]) + cdf2)
    if isinf(result) || isnan(result)
        return logprobOrderStatisticPoisson(Y,mu,D,j,precision=5*precision)
    else
        return Float64(result)
    end
end

function logpmfOrderStatPoisson(Y,mu,D,j)
    llik = logpdf(OrderStatistic(Poisson(mu), D, j), Y)
    if isinf(llik) || isnan(llik)
        llik = logprobOrderStatisticPoisson(Y,mu,D,j)
    end
    return llik
end