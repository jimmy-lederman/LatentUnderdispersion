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

function fit(model::MatrixMF, data; nsamples=1000, nburnin=200, nthin=5, initialize=true, initseed=1, mask=nothing, verbose=true, info=nothing,skipupdate=nothing,constantinit=nothing)
    #some checks
    Y_NM = data["Y_NM"]
    @assert size(Y_NM) == (model.N, model.M) "Incorrect data shape"
    @assert all(x -> x >= 0, Y_NM[:]) "Not all values in data are greater than or equal to 0"
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
                # println(state[var])
            end
            
        end
    end
    
    S = nburnin + nthin*nsamples
    if verbose
        prog = Progress(S, desc="Burnin+Samples...")
    end
    println("start")
    for s in 1:S
        # if s > 2500
        #     println(s)
        # end
        if s < nburnin/2 && !isnothing(skipupdate)
            ~, state = backward_sample(model, data, state, mask, skipupdate)
        else
            ~, state = backward_sample(model, data, state, mask)
        end
        if s > nburnin && mod(s,nthin) == 0
            push!(samplelist, state)
        end
        # if mod(s,100) == 0
        #     println(s)
        # end
        if verbose next!(prog) end
    end
    if verbose finish!(prog) end

    return samplelist
end

function gewekeTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nburnin=1000, nthin=5,info=nothing)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    # sample forward from the prior and likelihood
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

function evaluateInfoRate(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true, cols=nothing)
    
    S = size(samples)[1]
    I = 0 #total number of masked points
    llik0count = 0
    inforatetotal = 0
    if verbose
        prog = Progress(S, desc="calculating inforate")
    end
    for row in 1:model.N
        for col in 1:model.M
            if !isnothing(cols) && !(col in cols)
                continue
            end
            if !isnothing(mask) && mask[row,col]
                llikvector = Vector{Float64}(undef, S)
                haveusedbackup = false
                for s in 1:S
                    sample = samples[s]
                    # @assert 1 == 2
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    llikvector[s] = llik
                    # if llik == 0
                    #     llik0count += 1
                    # end
                end
                inforatetotal += logsumexpvec(llikvector) - log(S)
                I += 1
                if verbose next!(prog) end
            end
        end
    end
    #println("0 count: ", llik0count)
    if verbose finish!(prog) end
    return inforatetotal/I
end

function logAverageHeldoutProbs(model::MatrixMF, data, samples; info=nothing, mask=nothing, verbose=true)
    S = size(samples)[1]
    heldoutprobs = []
    @assert !isnothing(mask)
    if verbose
        prog = Progress(S, desc="calculating logprobs")
    end
    for row in 1:model.N
        for col in 1:model.M
            if !isnothing(mask) && mask[row,col]
                llikvector = Vector{Float64}(undef, S)
                for s in 1:S
                    sample = samples[s]
                    llik = evalulateLogLikelihood(model, sample, data, info, row, col)
                    #if isinf(llik) println(row, " ", col, " ", s) end
                    llikvector[s] = llik
                end
                push!(heldoutprobs, [logsumexpvec(llikvector) - log(S),row,col])
                if verbose next!(prog) end
            end
        end
    end
    if verbose finish!(prog) end
    return heldoutprobs
end