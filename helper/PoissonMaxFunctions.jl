#using Distributions
# using LogExpFunctions
# using PyCall
# sympy = pyimport("sympy")

function safeTrunc(dist,lower,upper;n=1)
    try
        return rand(Truncated(dist, lower, upper), n)
    catch e
        if dist isa Poisson
            if lower == 0
                try 
                    Fmax = poisson_cdf_precise(upper,mean(dist))
                    result = quantile(dist, rand(Uniform(0,Fmax), n))
                    if isinf(result)
                        return fill(upper,n)
                    else
                        return result
                    end
                catch e 
                    return fill(upper,n) 
                end
            elseif isinf(upper)
                try
                    Fmin = poisson_cdf_precise(lower,mean(dist))
                    result = quantile(dist, rand(Uniform(Fmin,1), n))
                    if isinf(result)
                        return fill(lower,n)
                    else
                        return result
                    end 
                catch e 
                    return fill(lower,n)
                end
            end
        else
            if lower == 0
                try 
                    Fmax = cdf(dist,upper)
                    result = quantile(dist, rand(Uniform(0,Fmax), n))
                    if isinf(result) || Fmax == 0
                        return fill(upper,n) 
                    elseif Fmax == 1
                        return rand(dist,n)
                    else 
                        return result
                    end
                catch e 
                    return fill(upper,n) 
                end
            elseif isinf(upper)
                try
                    Fmin = cdf(dist,lower)
                    result = quantile(dist, rand(Uniform(Fmin,1), n))
                    if isinf(result) || Fmin == 1
                        return fill(lower,n)
                    elseif Fmax == 0
                        return rand(dist,n)
                    else
                        return result
                    end 
                catch e 
                    return fill(lower,n)
                end
            end
        end
    end
end   

function probYatIteration(Y,i,dist)
    num = pdf(dist,Y)*cdf(dist, Y)^(i-1)
    denom = cdf(dist, Y)^i - cdf(dist, Y-1)^i
    # if isinf(num/denom)
    #     @assert 1 == 2
    #     println("num: ", num)
    #     println("denom: ", denom)
    #     println("cdf: ", cdf(dist, Y))
    #     println("pdf: ", pdf(dist, Y))
    # end
    return num/denom
end


function sampleIndex(Y,D,dist)
    #this is an approximation for stability; take out if
    #testing for correctness
    if pdf(dist, Y) < 10e-5 && Y > mean(dist)
        return rand(DiscreteUniform(1,D))
    end
    # if pdf(dist, Y) < 10e-5 && Y < mean(dist)
        
    # end
    index = 1
    b  = []
    totalmasstried = 0
    @views for d in D:-1:1
        b1 = [1-p for p in b]
        probY = probYatIteration(Y,d,dist)
        if isempty(b1)
            stopprobtemp = probY
        else
            stopprobtemp = probY .* prod(b1)
        end
        stopprob = stopprobtemp / (1-totalmasstried)
        if stopprob > 1 && abs(stopprob - 1) < 10e-5
            stopprob = 1
        end
        try
            stop = rand(Bernoulli(stopprob))
            if stop
                break
            end
            if d == 2
                index = D
                break
            end
            totalmasstried += stopprobtemp 
            push!(b, probY)
            index += 1
        catch ex
            println(probY)
            println(stopprobtemp)
            println(totalmasstried)
            println(stopprob)
            println("Y: ", Y, " dist: ", dist, " D: ", D)
            @assert 1 == 2
        end
        
    end
    # end
    return index
end

function sampleSumGivenMax(Y,D,dist)
    if Y == 0
        return 0
    end
    index = sampleIndex(Y,D,dist)
    sample1 = safeTrunc(dist, 0, Y - 1, n=index - 1)
    sample2 = safeTrunc(dist, 0, Y, n=D - index)
    beep = sum(sample1) + Y + sum(sample2)
    return beep

end


#code to compute MaxPoisson logpmf

function logprob(s,z,D)
    if s != 0
        return logsubexp(D*logcdf(Poisson(z),s),D*logcdf(Poisson(z),s-1))
    else
        return D*logcdf(Poisson(z),s)
    end
end

function logprobsymbolic(s,z,D)
    s1, z1 = sympy.symbols("s1 z1")
    log_gamma_reg_inc1 = D*sympy.log(sympy.uppergamma(s1, z1) / sympy.gamma(s1))
    s2, z2 = sympy.symbols("s2 z2")
    log_gamma_reg_inc2 = D*sympy.log(sympy.uppergamma(s2, z2) / sympy.gamma(s2))
    if s != 0
        #combine = sympy.log(sympy.exp(log_gamma_reg_inc1) - sympy.exp(log_gamma_reg_inc2))
        combine = log_gamma_reg_inc1 + sympy.log(1 - sympy.exp(log_gamma_reg_inc2 - log_gamma_reg_inc1))
        combine_result = combine.subs(s1, s+1)
        combine_result = combine_result.subs(z1, Int(round(z))) #rounding the mean is a (small) approximation
        combine_result = combine_result.subs(z2, Int(round(z))) #rounding the mean is a (small) approximation
        combine_result = combine_result.subs(s2, s)
        combine_result = sympy.N(combine_result, 64)
        combine_result = convert(Float64, combine_result)
    else #if data point is 0
        combine = log_gamma_reg_inc1
        combine_result = combine.subs(s1, s+1)
        combine_result = combine_result.subs(z1, z)
        combine_result = sympy.N(combine_result, 64)
        combine_result = convert(Float64, combine_result)
    end
    return combine_result
end

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