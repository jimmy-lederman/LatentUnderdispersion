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
    if Fmax == 0
        return rand(dist, D)
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
        return sum(sample1) + Y + sum(sample2)
    catch ex
        #the built in truncation does not work if mu is too different than Y
        sample1 = backupTruncation(index-1, Y-1, dist)
        sample2 = backupTruncation(D-index, Y, dist)
        return sum(sample1) + Y + sum(sample2)
    end
end