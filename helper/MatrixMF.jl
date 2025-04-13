using ProgressMeter
using Random 

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

abstract type MatrixMF end

function fit(model::MatrixMF, data; nsamples=1000, nburnin=200, nthin=5, initialize=true, initseed=1, mask=nothing, verbose=true, info=nothing,skipupdate=nothing,constantinit=nothing,griddy=false,annealStrat=nothing,firstiter=false)
    #some checks
    # Y_NM = data["Y_NM"]
    # @assert size(Y_NM) == (model.N, model.M) "Incorrect data shape"
    #@assert all(x -> x >= 0, Y_NM[:]) "Not all values in data are greater than or equal to 0"
    #housekeeping
    samplelist = []
    if initialize
        Random.seed!(initseed)
        if !isnothing(constantinit) && !isnothing(info)
            state = sample_prior(model, info, constantinit)
        elseif !isnothing(info)
            state = sample_prior(model, info)
        else
            state = sample_prior(model)
        end
        if !isnothing(constantinit)
            for (var, value) in constantinit
                state[var] = value
            end
            # println(state)
        end
    end
    if !isnothing(annealStrat)
        anneal = 1
    else
        anneal = nothing 
    end
    
    S = nburnin + nthin*nsamples
    if verbose
        prog = Progress(S, desc="Burnin+Samples...")
    end
    println("start")
    for s in 1:S
        # if s == 1 || s == 2
        #     println(state)
        # end
        if s == 1 && firstiter
            ~, state = backward_sample(model, data, state, mask, firstiter=true)
        end

        if s < nburnin/2 && !isnothing(skipupdate)
            ~, state = backward_sample(model, data, state, mask, skipupdate=skipupdate)
        elseif s < nburnin && !isnothing(annealStrat)#need to change to just nburnin later
            if s > nburnin/4 + anneal*(3*nburnin/4)/(model.D)
                anneal += 1
                println(anneal, " ", s)
            end
            @assert anneal <= model.D
            if s < nburnin/2 && griddy
                ~, state = backward_sample(model, data, state, mask, griddy=griddy,annealStrat=annealStrat,anneal=anneal)
            else
                ~, state = backward_sample(model, data, state, mask,annealStrat=annealStrat,anneal=anneal)
            end
        else
            if s < nburnin/2 && griddy
                ~, state = backward_sample(model, data, state, mask, griddy=true)
            else
                ~, state = backward_sample(model, data, state, mask)
            end
        end
        if s > nburnin && mod(s,nthin) == 0
            push!(samplelist, state)
        end
        # if mod(s,100) == 0
        #     println(s)
        # end
        if verbose next!(prog) end
        # println(s)
        flush(stdout)
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
            data, state = backward_sample(model, data, state)
        end
        for key in varlist
            push!(b_samples[key], state[key])
        end
    end
    return f_samples, b_samples
end

function scheinTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5, info=nothing)
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
            data, state = backward_sample(model, data, state)
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
                    inforatetotal += logsumexpvec(llikvector) - log(S)
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
    println("0 count: ", llik0count)
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

function logAverageHeldoutProbs(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true,sparse=true)
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