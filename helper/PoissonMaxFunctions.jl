using Distributions

function probYatIteration(Y,i,dist)
    num = pdf(dist,Y)*cdf(dist, Y)^(i-1)
    denom = cdf(dist, Y)^i - cdf(dist, Y-1)^i
    return num/denom
end


function sampleIndex(Y,D,dist)
    #this is an approximation for stability; take out if
    #testing for correctness
    if pdf(dist, Y) < 10e-3
        if Y > mean(dist)
            index = rand(DiscreteUniform(1,D))
        else
            index = 1
        end
    else
        index = 1
        b  = []
        totalmasstried = 0
        for d in D:-1:1
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
        end
    end
    return index
end

function backupTruncation(D, Y, dist)
    Fmax = cdf(dist, Y)
    if Fmax == 1
        return rand(dist, D)
    elseif Fmax == 0
        return fill(Y, D) #vector of all Y
    else
        u_n = rand(Uniform(0,Fmax), D)
        return quantile(dist, u_n)
    end
end

function sampleSumGivenMax(Y,D,dist)
    index = sampleIndex(Y,D,dist)
    try
        sample1 = rand(Truncated(dist, 0, Y - 1), index - 1)
        sample2 = rand(Truncated(dist, 0, Y), D - index)
        beep = sum(sample1) + Y + sum(sample2)
        # println("used good trunc")
        # println("Y: ", Y, " dist: ", dist, " D: ", D)
        return beep
    catch ex
        println("warning: using backup truncation")
        println("Y: ", Y, " dist: ", dist, " D: ", D)
        #the built in truncation does not work if mu is too different than Y
        sample1 = backupTruncation(index-1, Y-1, dist)
        sample2 = backupTruncation(D-index, Y, dist)
        return sum(sample1) + Y + sum(sample2)
    end
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
    result1 = log_gamma_reg_inc1.subs(s1, s+1)
    result1 = result1.subs(z1, Int(round(z)))
    result1 = convert(Float64, sympy.N(result1, 50))

    s2, z2 = sympy.symbols("s2 z2")
    log_gamma_reg_inc2 = D*sympy.log(sympy.uppergamma(s2, z2) / sympy.gamma(s2))
    result2 = log_gamma_reg_inc2.subs(z2, Int(round(z)))
    result2 = result2.subs(s2, s)
    result2 = convert(Float64, sympy.N(result2, 50))
    if s != 0
        return logsubexp(result1,result2)
    else
        return result1
    end
end

function logpmfMaxPoisson(Y,mu,D)
    try
        llik = logpdf(OrderStatistic(Poisson(mu), D, D), Y)
        if isinf(llik) || isnan(llik)
            llik = logprob(Y,mu,D)
        end
        return llik
    catch ex
        llik = logprobsymbolic(Y,mu,D)
        return llik
    end
end

