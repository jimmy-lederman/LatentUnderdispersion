using ProgressMeter

abstract type MatrixMF end

function fit(model::MatrixMF, data; nsamples=1000, nburnin=200, nthin=5, initialize=true, mask=nothing, verbose=true)
    #some checks
    Y_NM = data["Y_NM"]
    @assert size(Y_NM) == (model.N, model.M) "Incorrect data shape"
    @assert all(x -> x >= 0, Y_NM[:]) "Not all values in data are greater than or equal to 0"
    #housekeeping
    samplelist = []
    if initialize
        state = sample_prior(model)
    end
    S = nburnin + nsamples
    if verbose
        prog = Progress(S, desc="Burnin+Samples...")
    end
    for s in 1:S
        ~, state = backward_sample(model, data, state, mask)
        if s > nburnin && mod(s,nthin) == 0
            push!(samplelist, state)
        end
        if verbose next!(prog) end
    end
    if verbose finish!(prog) end

    return samplelist
end

function gewekeTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nburnin=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    # sample forward from the prior and likelihood
    @showprogress for i in 1:nsamples
        ~, state = forward_sample(model)
        #  we will only collect the states
        for key in varlist
            push!(f_samples[key], state[key])
        end
    end

    # first initialize with a forward sample of the state, data
    data, state = forward_sample(model)
    
    # burn-in samples
    @showprogress for i in 1:nburnin
        # perform a single Gibbs transition
        ~, state = backward_sample(model, data, state)
        # generate data from the new state
        data, ~ = forward_sample(model, state)
    end
    
    @showprogress for i in 1:(nsamples*nthin)
        # perform a single Gibbs transition
        ~, state = backward_sample(model, data, state)
        # generate data from the new state
        data, ~ = forward_sample(model, state)
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

function gandyscottTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    @showprogress for i in 1:nsamples
        #forward set
        data, state = forward_sample(model)
        for key in varlist
            push!(f_samples[key], state[key])
        end
        #backward set 
        data, state = forward_sample(model)
        for j in 1:nthin
            data, state = backward_sample(model, data, state)
        end
        for key in varlist
            push!(b_samples[key], state[key])
        end
    end
    return f_samples, b_samples
end

function scheinTest(model::MatrixMF, varlist::Vector{String}; nsamples=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    @showprogress for i in 1:nsamples
        #forward set
        data, state = forward_sample(model)
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

function evaluateInfoRate(model::MatrixMF, data, samples, mask=nothing)
    S = size(samples)[1]
    if !isnothing(mask)
        I = count(x -> x != 0, mask)
    else
        I = model.N*model.M
    end
    inforatetotal = 0
    for row in 1:model.N
        for col in 1:model.M
            if !isnothing(mask) && mask[row,col]
                pointtotal = 0
                for sample in samples

                    pointtotal += evaluateLikelihod(model, sample, data, row, col)
                end
                inforatetotal += log((1/S)*pointtotal)
            end
        end
    end
    return exp(inforatetotal/I)
end