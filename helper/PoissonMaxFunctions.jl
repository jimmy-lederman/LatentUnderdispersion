using Distributions
using LogExpFunctions
using PyCall
sympy = pyimport("sympy")


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
    if Y == 0
        return 0
    end
    index = sampleIndex(Y,D,dist)
    # try
    sample1 = rand(Truncated(dist, 0, Y - 1), index - 1)
    sample2 = rand(Truncated(dist, 0, Y), D - index)
    beep = sum(sample1) + Y + sum(sample2)
    # println("used good trunc")
    # println("Y: ", Y, " dist: ", dist, " D: ", D)
    return beep
    # catch ex
    #     println("warning: using backup truncation")
    #     println("Y: ", Y, " dist: ", dist, " D: ", D)
    #     #the built in truncation does not work if mu is too different than Y
    #     sample1 = backupTruncation(index-1, Y-1, dist)
    #     sample2 = backupTruncation(D-index, Y, dist)
    #     return sum(sample1) + Y + sum(sample2)
    # end
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
        if isinf(llik) || isnan(llik)
            llik = logprob(Y,mu,D)
        end
        if isinf(llik) || isnan(llik)
            llik = logprobsymbolic(Y,mu,D)
        end
        return llik
    catch ex
        llik = logprobsymbolic(Y,mu,D)
        return llik
    end
end


function prob(Y,D,j,dist)
    c = pdf(OrderStatistic(dist, D, j), Y)
    if D == j == 1
        return [0,1,0]
    end
    if j == 1
        B = 0
    else
        B = cdf(dist,Y-1)*pdf(OrderStatistic(dist, D-1, j-1), Y)/c #probability of z1 < y
    end
    if D == j
        A = 0
    else 
        A = ccdf(dist,Y)*pdf(OrderStatistic(dist, D-1, j), Y)/c #probability of z1 > y
    end
    if isnan(A) || isnan(B)
        println(c)
        println(cdf(dist,Y-1)*pdf(OrderStatistic(dist, D-1, j-1), Y))
        println(ccdf(dist,Y)*pdf(OrderStatistic(dist, D-1, j), Y))
        println("Y: ", Y, " ", dist)
        println([B,1-A-B,A])
        @assert 1 == 2
    end
    return [B,1-A-B,A]
end

function sampleFirstMax(Y,D,dist)
    if pdf(dist, Y) < 10e-5 && Y > mean(dist)
        probY = 1/D
    end 
    probY = probYatIteration(Y, D, dist)
    c = rand(Bernoulli(probY))
    if c == 1 #Z1 = Y
        return Y 
    else #Z1 < Y
        return rand(Truncated(dist, 0, Y-1))
    end
end