
#code to compute MaxPoisson logpmf

function logprob(s,z,D)
    if s != 0
        return logsubexp(D*logcdf(Poisson(z),s),D*logcdf(Poisson(z),s-1))
    else
        return D*logcdf(Poisson(z),s)
    end
end

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
    try
        llik = logpdf(OrderStatistic(Poisson(mu), D, D), Y)
        # if isinf(llik) || isnan(llik)
        #     llik = logprob(Y,mu,D)
        # end
        if isinf(llik) || isnan(llik)
            llik = logprobsymbolic(Y,mu,D)
        end
        return llik
    catch ex
        llik = logprobsymbolic(Y,mu,D)
        return llik
    end
end