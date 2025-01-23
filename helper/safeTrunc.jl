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