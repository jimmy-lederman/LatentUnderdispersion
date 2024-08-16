using Distributions

function pmfDposterior(d, r, p, Y, dist)
    num = binomial(d+r-1,d)*(1-p)^d*(cdf(dist, Y)^d - cdf(dist, Y-1)^d)
    denom = 1/(1- (1-p)*cdf(dist, Y))^r - 1/(1- (1-p)cdf(dist, Y-1))^r
    result = num/denom
    if isinf(result) || isnan(result)
        result = binomial(d+r-1,d)*(1-p)^(d-1) * p^(r+1) * (d/r)
    end
    return result
end

function sampleDposterior(r,p,Y,dist)
    D = 1
    totalmasstried = 0
    while true
        stopprobtemp = pmfDposterior(D,r,p,Y,dist)
        stopprob = stopprobtemp/(1-totalmasstried)
        if isnan(stopprob) || isinf(stopprob) || stopprob < 0 || stopprob > 1 
            println(stopprob, " ",stopprobtemp, " ", totalmasstried)
            println("Y: ", Y, " dist: ", dist)
        end
        if (stopprob > 1 && abs(stopprob - 1) < 10e-3) || rand(Bernoulli(stopprob))
            break
        end
        D += 1
        totalmasstried += stopprobtemp
    end
    return D
end

function pmfDposteriorMedian(d, r, p, Y, dist)
    F_Y = cdf(dist, Y)
    F_Ym1 = cdf(dist, Y-1)
    num = binomial(d+r-1,d)*(1-p)^d*((F_Y*(1-F_Y))^(d/2)*(F_Y/(1-F_Y))^(1/2) - (F_Ym1*(1-F_Ym1))^(d/2)*(F_Ym1/(1-F_Ym1))^(1/2))
    denom = ((F_Y/(1-F_Y))^(1/2))*(1/(1- (1-p)*sqrt(F_Y*(1-F_Y)))^r - 1) - ((F_Ym1/(1-F_Ym1))^(1/2))*(1/(1- (1-p)*sqrt(F_Ym1*(1-F_Ym1)))^r - 1)
    result = num/denom
    # if isinf(result) || isnan(result)
    #     result = binomial(d+r-1,d)*(1-p)^(d-1) * p^(r+1) * (d/r)
    # end
    return result
end

r0 = 1
p0 = .5
mu = 20
Y = 21
beep = [pmfDposterior(d, r0, p0, Y, Poisson(mu)) for d in 1:30]
println(beep[1:30])
println(sum(beep))
beep = [pmfDposteriorMedian(d, r0, p0, Y, Poisson(mu)) for d in 1:30]
println(beep[1:30])
println(sum(beep))

# # beep = [pmfDposterior(d, r0, p0, Y, Poisson(mu)) for d in 1:20]
# # println(beep[1:5])
# # println(sum(beep))

# function pmfconstant(d, r, p)
#     num = binomial(d+r-1,d)*(1-p)^(d)*d/r*(p)^r * (p)/(1-p)
#     return num
# end

# beep = [pmfconstant(d, r0, p0) for d in 1:20]
# println(beep[1:5])
# println(sum(beep))