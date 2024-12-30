using Distributions

function safeTrunc(dist,lower,upper;n=1)
    try
        return rand(Truncated(dist, lower, upper), n)
    catch e
        println("beep")
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

function probYatIterationMin(Y,i,dist)
    num = pdf(dist,Y)*ccdf(dist, Y-1)^(i-1)
    denom = ccdf(dist, Y-1)^i - ccdf(dist, Y)^i
    if isnan(num/denom)
        println(dist)
        println(Y)
        println(i)
        println(num)
        println(denom)
        @assert 1 == 2
        
    end
    return num/denom
end


function sampleIndexMin(Y,D,dist)
    #this is an approximation for stability; take out if
    #testing for correctness
    # if pdf(dist, Y) < 10e-5 && Y > mean(dist)
    #     return 1
    # end
    if pdf(dist, Y) < 10e-10 && Y < mean(dist)
        return rand(DiscreteUniform(1,D))
    end
    if pdf(dist, Y) < 10e-20 && Y > mean(dist)
        return 1
    end
    index = 1
    b  = []
    totalmasstried = 0
    @views for d in D:-1:1
        b1 = [1-p for p in b]
        probY = probYatIterationMin(Y,d,dist)
        if isempty(b1)
            stopprobtemp = probY
        else
            stopprobtemp = probY .* prod(b1)
        end
        stopprob = stopprobtemp / (1-totalmasstried)
        try
            # if stopprob > 1 && abs(stopprob - 1) < 10e-5
            #     stopprob = 1
            # end
            stop = rand(Bernoulli(stopprob))
            if stop
                break
            end
        catch ex
            println(Y, " ", D, " ", dist, " ", mean(dist))
            println(stopprob)
            println(probY)
            @assert 1 == 2
        end
        if d == 2
            index = D
            break
        end
        totalmasstried += stopprobtemp 
        push!(b, probY)
        index += 1
    end
    # end
    return index
end


function sampleSumGivenMin(Y,D,dist)
    index = sampleIndexMin(Y,D,dist)
    # println("i: ", index)
    sample1 = safeTrunc(dist, Y + 1, Inf, n=index - 1)
    sample2 = safeTrunc(dist, Y, Inf, n=D - index)
    beep = sum(sample1) + Y + sum(sample2)
    # println("sum: ", beep)
    return beep
end