using ProgressMeter
using Random
using Distributions
using JLD          # only for fit's optional checkpoint/resume; already a Project dependency


function logsumexpvec(arr::AbstractVector{T}) where T <: Real
    m = maximum(arr)
    m + log(sum(exp.(arr .- m)))
end

function gewekepvalue(forw, back)
    num = mean(forw) - mean(back)
    varf = mean([u^2 for u in forw]) - mean(forw)^2
    varb = mean([u^2 for u in back]) - mean(back)^2
    teststat = num/((1/length(forw))*varf+(1/length(back))*varb)^(1/2)
    return 2*(1 - cdf(Normal(0,1), abs(teststat)))
end

# Thin a count z across K components with weights P_K (need not be normalized),
# writing the result into a preallocated buffer.
#
# This is the standard conditional-binomial construction of a multinomial and is
# EXACT IN DISTRIBUTION -- identical to rand(Multinomial(z, P_K ./ sum(P_K))), which
# Distributions implements the same way internally. It differs only in consuming the
# RNG stream directly rather than through a freshly constructed distribution object,
# and it avoids three allocations per call (the normalized weight vector, the
# Multinomial object, and the output vector).
function thin_multinomial!(buf::AbstractVector{Int}, z::Integer, P_K::AbstractVector{Float64}, K::Integer)
    rem = Int(z)
    s = 0.0
    @inbounds for k in 1:K
        s += P_K[k]
    end
    @inbounds for k in 1:(K-1)
        if rem <= 0 || s <= 0
            buf[k] = 0
        else
            q = P_K[k] / s
            q = q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q)
            zk = rand(Binomial(rem, q))
            buf[k] = zk
            rem -= zk
        end
        s -= P_K[k]
    end
    @inbounds buf[K] = rem > 0 ? rem : 0
    return buf
end

# --- hierarchical rate for the Gamma factor prior --------------------------
# Models put V_km ~ Gamma(c, d). Holding d fixed lets the prior, rather than the
# data, choose the scale of V. That is harmless where UV is the mean (Poisson,
# median-Poisson, STAR-NN: the likelihood pins the scale) but decisive where UV
# is the negative-binomial size and the mean is r(1-p)/p -- there the scale and
# the dispersion trade off along a ridge, and a fixed d silently picks the split,
# biasing p (measured: p ~ 0.32 against a true 0.25 on the Section 6.3 NB data).
#
# Giving d its own Gamma(e0, f0) prior is conjugate in every one of these models:
#     d | V ~ Gamma(e0 + n_V * c,  f0 + sum(V))
# so it costs one extra draw per sweep. Applied uniformly so the prior
# specification is the same sentence for every model.
const HYPER_E0 = 1.0
const HYPER_F0 = 1.0

sample_rate_hyper(V, c) =
    rand(Gamma(HYPER_E0 + length(V) * c, 1 / (HYPER_F0 + sum(V))))

# initial draw, used by sample_prior so chains start dispersed in d as well
sample_rate_prior() = rand(Gamma(HYPER_E0, 1 / HYPER_F0))

abstract type MatrixMF end

function fit(model::MatrixMF, data; nsamples=1000, nburnin=200, nthin=5, initialize=true, initseed=1, mask=nothing, verbose=true, info=nothing,
    constantinit=nothing, init=nothing, keep_all=false, skipupdate=nothing,
    griddy=false, S_griddy=250, annealStrat=nothing,
    ckptfile=nothing, ckptevery=500)
    # `skipupdate` names parameters whose update block should not RUN at all, and is
    # forwarded to backward_sample. It is distinct from `constantinit`, which lets the
    # update run and then overwrites the result: the two give the same chain, but only
    # skipupdate makes a pinned parameter cheap. That matters where the skipped block is
    # a large share of a sweep -- for the covid model the D branch (the D update, the
    # Polya-Gamma draws and the beta/tau blocks) is about half of one, so a fixed-D run
    # scored on wall-clock would otherwise be charged for machinery it does not use.
    # Defaults to nothing, in which case backward_sample is called exactly as before, so
    # models whose backward_sample has no such keyword are unaffected.
    # `init` SEEDS a parameter and then lets it move; `constantinit` PINS it for the whole
    # run. Both are needed: the fixed-D columns pin D, while the median-NB model needs a
    # data-driven starting point for r (its prior sits at r ~ 0.01 against a needed ~500,
    # and from a prior draw the latent sums overflow Int64 before the hierarchical rate
    # can adapt) without freezing r thereafter.
    # Initialization is a draw from the prior for every model, except that
    # `griddy=true` PREPENDS S_griddy iterations in which (p, r) are drawn from a
    # grid over p rather than from their complete conditionals. Those iterations
    # precede burn-in and are never retained.
    #
    # This branch was removed in ea4a13e as a workaround for a p update that never
    # ran, and restored here because the Section 6.2 parameter-recovery results
    # (Feb 2026) were produced WITH it: alpha and p are strongly correlated under
    # this parameterization, and from a prior draw the chain reaches the right
    # neighbourhood of (alpha, p) only slowly. Without it the repo cannot
    # reproduce Figure 7. Default is false, so Section 6.3 and every other caller
    # is unaffected.
    #
    # `annealStrat` is restored alongside griddy for the same reason: the Section
    # 6.2 notebook passes it, and it is NOT inert -- the univariate NB model uses
    # it to set shape_factor/rate_factor, which scale the sufficient statistics in
    # the (mu, p) updates. It ramps `anneal` down from annealstart to 0 over the
    # first half of burn-in in annealtimes steps. Default nothing, so every other
    # caller is unaffected.
    #
    # The skipupdate branch (parameters frozen while s < nburnin/2) stays removed:
    # holding D fixed early is unnecessary (verified: R-hat <= 1.004 for the
    # median-Poisson model with D free from the first iteration). `constantinit`
    # is kept because it is how a model is pinned to a fixed order D.
    samplelist = []
    s0 = 0
    # ---- optional checkpoint / resume -------------------------------------------------
    # Opt-in: with ckptfile = nothing nothing is written or read, so every other section's
    # behaviour is untouched. It exists because the Section 7.2 held-out runs are 8-10 h on
    # the full matrix against a 12 h wall clock, so a job that is cut off can be resubmitted
    # and continue instead of starting over.
    #
    # CAVEAT: a resumed chain is NOT bitwise identical to an uninterrupted one. The sampler
    # draws from per-task RNGs inside @threads loops and those cannot be captured, so this is
    # crash recovery, not reproducibility. (The same caveat applies to run_diag_chain.jl's
    # own resume path.) The retained draws are still a valid sample from the chain either way.
    if !isnothing(ckptfile) && isfile(ckptfile) && filesize(ckptfile) > 0
        ck = JLD.load(ckptfile)
        state = ck["state"]; s0 = ck["sweep"]; samplelist = ck["samplelist"]
        initialize = false
        verbose && println("resuming from $ckptfile at sweep $s0 ($(length(samplelist)) draws retained)")
        flush(stdout)
    end
    if initialize
        Random.seed!(initseed)
        if !isnothing(constantinit) && !isnothing(info)
            state = sample_prior(model, info, constantinit)
        elseif !isnothing(info)
            state = sample_prior(model, info)
        else
            state = sample_prior(model)
        end
        if !isnothing(init)
            for (var, value) in init
                # A Function is called on the (already prior-drawn) state, so a seed can
                # depend on the draw rather than replace it. That is what lets a chain keep
                # the dispersion of its own prior draw while having only its scale
                # corrected -- see the covid run scripts, where the prior's scale for phi is
                # wrong by three orders of magnitude and a fixed seed would make every chain
                # start at the same point.
                state[var] = value isa Function ? value(state) : value
            end
        end
        if !isnothing(constantinit)
            for (var, value) in constantinit
                state[var] = value
            end
        end
    end

    # griddy iterations sit in FRONT of burn-in, so nburnin and nsamples keep
    # their usual meaning and the retained draws are unchanged in number.
    Sg = griddy ? S_griddy : 0
    S = Sg + nburnin + nthin*nsamples
    if verbose
        prog = Progress(S, desc="Burnin+Samples...")
    end

    annealstart = 1000
    annealtimes = 10
    anneal = isnothing(annealStrat) ? nothing : annealstart
    annealcount = 1

    for s in (s0+1):S
        # anneal ramps down over the first half of burn-in (measured after any
        # griddy prefix), matching the pre-ea4a13e schedule.
        if !isnothing(annealStrat) && s > Sg && (s - Sg) < nburnin/2
            if (s - Sg) > nburnin*annealcount/(annealtimes*2)
                anneal -= annealstart/annealtimes
                annealcount += 1
            end
        elseif !isnothing(annealStrat) && s > Sg
            anneal = nothing
        end

        ~, state = if griddy && s <= Sg
            backward_sample(model, data, state, mask; griddy=true,
                            annealStrat=annealStrat, anneal=anneal)
        elseif !isnothing(annealStrat)
            backward_sample(model, data, state, mask;
                            annealStrat=annealStrat, anneal=anneal)
        elseif isnothing(skipupdate)
            backward_sample(model, data, state, mask)
        else
            backward_sample(model, data, state, mask; skipupdate=skipupdate)
        end

        # `constantinit` PINS these entries, it does not merely seed them. Models sample
        # every parameter unconditionally inside backward_sample, so without this a
        # "fixed D" run would silently become a D-inferred run: constantinit would set
        # D once and the model's own D update would immediately overwrite it. Restoring
        # after each sweep is what makes the fixed-D columns of the flights Table 2
        # (D = 1, 3, 5, 7, 9) actually hold D fixed. Cost is one dictionary assignment
        # per pinned variable per sweep; the update that produced the discarded value
        # still runs, which for the flights model is ~6% of an iteration.
        if !isnothing(constantinit)
            for (var, value) in constantinit
                state[var] = value
            end
        end

        if keep_all
            push!(samplelist, state)
        elseif s > Sg + nburnin && mod(s - Sg, nthin) == 0
            push!(samplelist, state)
        end

        if !isnothing(ckptfile) && mod(s, ckptevery) == 0
            # write to a temporary and move, so a job killed mid-write does not leave a
            # truncated checkpoint that the next attempt would try to resume from
            tmp = ckptfile * ".tmp"
            JLD.save(tmp, "state", state, "sweep", s, "samplelist", samplelist)
            mv(tmp, ckptfile; force=true)
        end

        if verbose
            next!(prog)
            flush(stdout)
        end
    end
    if verbose finish!(prog) end

    return samplelist
end

function gewekeTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nburnin=1000, nthin=5,info=nothing)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    #sample forward from the prior and likelihood
    @showprogress for i in 1:nsamples
        ~, state = forward_sample(model,info=info)
        #  we will only collect the states
        for key in varlist
            push!(f_samples[key], state[key])
        end
    end

    # first initialize with a forward sample of the state, data
    data, state = forward_sample(model, info=info)
    
    # burn-in samples
    @showprogress for i in 1:nburnin
        # perform a single Gibbs transition
        ~, state = backward_sample(model, data, state)
        # generate data from the new state
        data, ~ = forward_sample(model, state=state, info=info)
    end
    
    @showprogress for i in 1:(nsamples*nthin)
        # perform a single Gibbs transition
        ~, state = backward_sample(model, data, state)
        # generate data from the new state
        data, ~ = forward_sample(model, state=state, info=info)

        # condition on the new data by redefining the transition operator
        # collect every n_thin sample
        if mod(i, nthin) == 0
            # we will only collect the states
            for key in varlist
                push!(b_samples[key], state[key])
            end
        end
    end
    return f_samples, b_samples
end

function gandyscottTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5, info=nothing)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    @showprogress for i in 1:nsamples
        #forward set
        data, state = forward_sample(model, info)
        for key in varlist
            push!(f_samples[key], state[key])
        end
        #backward set 
        data, state = forward_sample(model, info)
        for j in 1:nthin
            # Skipping a block is still a valid (partial) Gibbs kernel, so the test stays
            # exact; it lets one isolate a subset of the updates.
            data, state = isnothing(skipupdate) ? backward_sample(model, data, state) :
                                                  backward_sample(model, data, state; skipupdate=skipupdate)
        end
        for key in varlist
            push!(b_samples[key], state[key])
        end
    end
    return f_samples, b_samples
end

function scheinTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5, info=nothing, skipupdate=nothing)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    @showprogress for i in 1:nsamples
        #forward set
        data, state = forward_sample(model, info=info)
        for key in varlist
            push!(f_samples[key], state[key])
        end
        #backward set 
        for j in 1:nthin
            # Skipping a block is still a valid (partial) Gibbs kernel, so the test stays
            # exact; it lets one isolate a subset of the updates.
            data, state = isnothing(skipupdate) ? backward_sample(model, data, state) :
                                                  backward_sample(model, data, state; skipupdate=skipupdate)
        end
        for key in varlist
            push!(b_samples[key], state[key])
        end
    end
    return f_samples, b_samples
end

function sequential_test(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5, iterations=10, alpha=0.05, delta = 10, funcs = [Statistics.var, geometric_mean], info=nothing,test="schein")
    beta = alpha/iterations
    nu = beta^(1/iterations)
    
    for i in 1:iterations
        println(nu + beta)
        if test == "schein"
            fsamples, bsamples = scheinTest(model,varlist,nsamples=nsamples,nthin=nthin,info=info)
        elseif test == "geweke"
            fsamples, bsamples = gewekeTest(model,varlist,nsamples=nsamples,nthin=nthin,info=info)
        else
            throw("no test called that dummy")
        end
        pvals = Float64[]
        for key in keys(fsamples)
            if fsamples[key][1] isa Number
                push!(pvals, gewekepvalue(fsamples[key], bsamples[key]))
            else
                for func in funcs
                    push!(pvals, gewekepvalue([func(s) for s in fsamples[key]], [func(s) for s in bsamples[key]]))
                end
            end
        end
        q = minimum(pvals)*length(pvals)
        if q <= beta
            return "fail", fsamples, bsamples
        elseif q > nu + beta 
            return "OK", fsamples, bsamples
        else 
            beta = beta/nu
            nsamples = nsamples*delta
        end
    end
    return "ended", fsamples, bsamples
end

function evaluateInfoRate(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true, cols=nothing,sparse=false)
    S = size(samples)[1]
    I = 0 #total number of masked points
    llik0count = 0
    inforatetotal = 0
    #badmat = zeros(39,3)
    if verbose
        prog = Progress(S, desc="calculating inforate")
    end
    # println(S)
    if !sparse
        @views for row in 1:model.N
            @views for col in 1:model.M
                if !isnothing(cols) && !(col in cols)
                    continue
                end
                if !isnothing(mask) && mask[row,col]
                    llikvector = Vector{Float64}(undef, S)
                    for s in 1:S
                        sample = samples[s]
                        llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                        llikvector[s] = llik
                    end
                    # logsumexpvec was previously evaluated twice per heldout cell
                    # (@assert is enabled by default in Julia, so the assertion copy
                    # was not free); compute once and reuse.
                    lse = logsumexpvec(llikvector)
                    @assert !isnan(lse)
                    inforatetotal += lse - log(S)
                    I += 1
                    if verbose next!(prog) end
                end
            end
            #if row % 1000 == 0 println(row) end
        end
    else #sparse
        @views for ind in axes(Ysparse, 1)
            count = Ysparse[ind, :]
            col = count[1]
            row = count[2]  
            if !isnothing(cols) && !(col in cols)
                continue
            end
            if !isnothing(mask) && mask[ind]
                llikvector = Vector{Float64}(undef, S)
                haveusedbackup = false
                for s in 1:S
                    sample = samples[s]
                    # @assert 1 == 2
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    llikvector[s] = llik
                    # if llik == 0
                    #     llik0count += 1
                    #     badmat[llik0count,:] = [data["Y_NM"][row,col],dot(sample["U_NK"][row,:], sample["V_KM"][:,col]),sample["p_N"][row]]
                    # end
                end
                inforatetotal += logsumexpvec(llikvector) - log(S)
                I += 1
                if verbose next!(prog) end
            end
        end
    end
    # println(total)
    # flush(stdout)
    # @assert 1 == 2
    if llik0count > 0 println("0 count: ", llik0count) end
    if verbose finish!(prog) end
    return inforatetotal/I#, badmat
end

function evaluateInfoRateCategories(model::MatrixMF, data, samples, categories; info=nothing, mask=nothing, verbose=true, cols=nothing,sparse=false)
    I = zeros(length(unique(categories))) #total number of masked points
    inforatetotal = zeros(length(unique(categories)))
    # if verbose
    #     prog = Progress(S, desc="calculating inforate")
    # end
    @views for row in 1:model.N
        @views for col in 1:model.M
            if !isnothing(cols) && !(col in cols)
                continue
            end
            if !isnothing(mask) && mask[row,col]                
                sample = samples[1]
                llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                cat = categories[row,col]
                inforatetotal[cat] += llik
                I[cat] += 1
                #if verbose next!(prog) end
            end
        end
        #if row % 1000 == 0 println(row) end
    end


 

    #f verbose finish!(prog) end
    return inforatetotal ./ I#, badmat
end

function evaluateInfoRateSplit(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true, cols=nothing, cutoff=0)
    S = size(samples)[1]
    I1 = 0 #total number of masked points
    inforatetotal1 = 0
    I2 = 0
    inforatetotal2 = 0
    if verbose
        prog = Progress(S, desc="calculating inforate")
    end
    @views for row in 1:model.N
        @views for col in 1:model.M
            if !isnothing(cols) && !(col in cols)
                continue
            end
            if !isnothing(mask) && mask[row,col]
                llikvector = Vector{Float64}(undef, S)
                for s in 1:S
                    sample = samples[s]
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    llikvector[s] = llik
                end
                if data["Y_NM"][row,col] <= cutoff
                    inforatetotal1 += logsumexpvec(llikvector) - log(S)
                    I1 += 1
                else
                    inforatetotal2 += logsumexpvec(llikvector) - log(S)
                    I2 += 1
                end
                if verbose next!(prog) end
            end
        end
    end
    if verbose finish!(prog) end
    return [inforatetotal1/I1, inforatetotal2/I2]
end

function evaluateInfoRateSplit3(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true, cols=nothing, cutoff1=0, cutoff2=0)
    S = size(samples)[1]
    I1 = 0 #total number of masked points
    inforatetotal1 = 0
    I2 = 0
    inforatetotal2 = 0
    I3 = 0
    inforatetotal3 = 0
    if verbose
        prog = Progress(S, desc="calculating inforate")
    end
    @views for row in 1:model.N
        @views for col in 1:model.M
            if !isnothing(cols) && !(col in cols)
                continue
            end
            if !isnothing(mask) && mask[row,col]
                llikvector = Vector{Float64}(undef, S)
                for s in 1:S
                    sample = samples[s]
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    llikvector[s] = llik
                end
                if data["Y_NM"][row,col] <= cutoff1
                    inforatetotal1 += logsumexpvec(llikvector) - log(S)
                    I1 += 1
                elseif data["Y_NM"][row,col] <= cutoff2
                    inforatetotal2 += logsumexpvec(llikvector) - log(S)
                    I2 += 1
                else 
                    inforatetotal3 += logsumexpvec(llikvector) - log(S)
                    I3 += 1
                end
                if verbose next!(prog) end
            end
        end
    end
    if verbose finish!(prog) end
    return [inforatetotal1/I1, inforatetotal2/I2, inforatetotal3/I3]
end

function logAverageHeldoutProbs(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true,sparse=false)
    S = size(samples)[1]
    heldoutprobs = []
    total = 0
    @assert !isnothing(mask)
    if verbose
        prog = Progress(S, desc="calculating logprobs")
    end
    if !sparse
        for row in 1:model.N
            for col in 1:model.M
                if !isnothing(mask) && mask[row,col]
                    llikvector = Vector{Float64}(undef, S)
                    for s in 1:S
                        sample = samples[s]
                        llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                        #if isinf(llik) println(row, " ", col, " ", s) end
                        if llik == 0
                            total += 1
                        end
                        llikvector[s] = llik
                    end
                    push!(heldoutprobs, [logsumexpvec(llikvector) - log(S),row,col])
                    if verbose next!(prog) end
                end
            end
        end
    else #sparse
        @views for ind in axes(Ysparse, 1)
            count = Ysparse[ind, :]
            col = count[1]
            row = count[2]  
            if !isnothing(mask) && mask[ind]
                llikvector = Vector{Float64}(undef, S)
                haveusedbackup = false
                for s in 1:S
                    sample = samples[s]
                    # @assert 1 == 2
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    llikvector[s] = llik
                    # if llik == 0
                    #     llik0count += 1
                    #     badmat[llik0count,:] = [data["Y_NM"][row,col],dot(sample["U_NK"][row,:], sample["V_KM"][:,col]),sample["p_N"][row]]
                    # end
                end
                push!(heldoutprobs, [logsumexpvec(llikvector) - log(S),row,col])
                if verbose next!(prog) end
            end
        end
    end

    if verbose finish!(prog) end
    return heldoutprobs
end