using StatsBase

"""
Compute ESS for multiple parameters and chains.
chains: Array{Float64,3} with dims (iters, nparams, nchains)
maxlag: maximum lag for autocorrelation
"""
function ess_multichain(chains::Array{Float64,3}; maxlag::Int=500)
    (N, nparam, nch) = size(chains)
    ess_vec = zeros(nparam)

    for p in 1:nparam
        # flatten chains for this parameter (concatenate along iteration axis)
        combined = vec(permutedims(chains[:, p, :], (1,2)))  # length N*nch

        # compute ESS using Geyer initial positive sequence
        y = combined .- mean(combined)
        L = min(maxlag, length(y)-1)
        lags = 0:L
        ac = autocor(y, lags)

        s = 0.0
        k = 1
        while k < L
            pair = ac[k+1] + ac[k+2]  # 1-based indexing
            if pair <= 0
                break
            end
            s += pair
            k += 2
        end

        ess_val = length(y) / (1 + 2*s)
        ess_vec[p] = max(ess_val, 1.0)
    end

    return ess_vec
end

"""
Compute Effective Sample Size (ESS) for a univariate MCMC chain.
x: vector of samples
maxlag: maximum lag to consider (default 500)
Uses Geyer’s initial positive sequence rule.
"""
function ess(x::AbstractVector; maxlag::Int=500)
    n = length(x)
    y = x .- mean(x)
    L = min(maxlag, n - 1)
    lags = 0:L

    ac = autocor(y, lags)  # ac[1] = lag0 = 1.0
    s = 0.0
    k = 1
    while k < L
        pair = ac[k+1] + ac[k+2]  # add in pairs
        if pair <= 0
            break
        end
        s += pair
        k += 2
    end

    ess_val = n / (1 + 2*s)
    return max(ess_val, 1.0)  # guard against <1
end
#println(ess(estimates1))

"""
Compute split-chain R̂ for a single parameter.
chains: Array{Float64,3} with dims (iters, nparams=1, nchains)
"""
function rhat_split_univariate(chains::Array{Float64,3})
    (N, _, M) = size(chains)
    half = fld(N, 2)  # integer division

    # handle edge case
    if half < 2
        error("Chains too short to split")
    end

    # create 2*M split chains
    splits = Array{Float64,2}(undef, half, 2*M)
    for m in 1:M
        splits[:, 2*m-1] = chains[1:half, 1, m]
        splits[:, 2*m]   = chains[half+1:2*half, 1, m]
    end

    means = [mean(splits[:,j]) for j in 1:(2*M)]
    vars = [var(splits[:,j]; corrected=true) for j in 1:(2*M)]

    W = mean(vars)
    B = half / (2*M - 1) * sum((means .- mean(means)).^2)
    var_hat = (half - 1)/half * W + B/half

    rhat = sqrt(max(var_hat / W, 1.0))  # enforce R̂ ≥ 1
    return rhat
end

function rhat_split_multivariate(chains::Array{Float64,3})
    (N, nparams, M) = size(chains)
    half = fld(N, 2)

    if half < 2
        error("Chains too short to split")
    end

    rhat_vec = zeros(nparams)

    for p in 1:nparams
        # Create 2*M split chains for this parameter
        splits = Array{Float64,2}(undef, half, 2*M)
        for m in 1:M
            splits[:, 2*m-1] = chains[1:half, p, m]
            splits[:, 2*m]   = chains[half+1:2*half, p, m]
        end

        # Compute mean and variance per split chain
        means = [mean(splits[:, j]) for j in 1:(2*M)]
        vars  = [var(splits[:, j]; corrected=true) for j in 1:(2*M)]

        # Within-chain and between-chain variance
        W = mean(vars)
        B = half / (2*M - 1) * sum((means .- mean(means)).^2)

        # Posterior variance estimate
        var_hat = (half - 1)/half * W + B/half

        # Split-chain R̂, enforce ≥ 1
        rhat_vec[p] = sqrt(max(var_hat / W, 1.0))
    end

    return rhat_vec
end