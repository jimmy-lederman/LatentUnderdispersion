using Distributions

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
    #try
    sample1 = rand(Truncated(dist, Y + 1, Inf), index - 1)
    sample2 = rand(Truncated(dist, Y, Inf), D - index)
    beep = sum(sample1) + Y + sum(sample2)
    return beep
    # catch ex
    #     # println("warning: using backup truncation")
    #     # println("Y: ", Y, " dist: ", dist, " D: ", D)
    #     # #the built in truncation does not work if mu is too different than Y
    #     # sample1 = backupTruncation(index-1, Y-1, dist)
    #     # sample2 = backupTruncation(D-index, Y, dist)
    #     # return sum(sample1) + Y + sum(sample2)
    #     @assert 1 == 2
    # end
end