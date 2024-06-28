using ProgressBars

abstract type matrixMF end

function fit(model::matrixMF, data; nsamples=1000, nburnin=200, nthin=5, initialize=true, mask=nothing)
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
    for s in ProgressBar(1:S)
        data, state = backward_sample(model, data, state, mask)
        if s > nburnin && mod(s,nthin) == 0
            push!(samplelist, state)
        end
    end
    return samplelist
end

function gewekeTest(model::matrixMF, varlist::Vector{String}; nsamples=1000, nburnin=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    # sample forward from the prior and likelihood

    for i in ProgressBar(1:nsamples)
        ~, state = forward_sample(model)
        #  we will only collect the states
        for key in varlist
            push!(f_samples[key], state[key])
        end
    end

    # first initialize with a forward sample of the state, data
    data, state = forward_sample(model)
    
    # burn-in samples
    for i in ProgressBar(1:nburnin)
        # perform a single Gibbs transition
        ~, state = backward_sample(model, data, state)
        # generate data from the new state
        data, ~ = forward_sample(model, state)
    end
    
    for i in ProgressBar(1:(nsamples*nthin))
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

function gandyscottTest(model::matrixMF, varlist::Vector{String}; nsamples=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    for i in ProgressBar(1:nsamples)
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

function scheinTest(model::matrixMF, varlist::Vector{String}; nsamples=1000, nthin=5)
    f_samples = Dict("$key" => [] for key in varlist)
    b_samples = Dict("$key" => [] for key in varlist)
    for i in ProgressBar(1:nsamples)
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

function evaluateInfoRate(model::matrixMF, data, samples, mask=nothing)
    S = size(samples)[1]
    if !isnothing(mask)
        I = countnz(mask)
    else
        I = model.N*model.M
    end
    inforatetotal = 0
    L_NMS = zeros(S, model.N, model.M)
    for (i, state) in enumerate(samples)
        L_NMS[i, :, :] .= evaluateLikelihod(model, state, data, mask)
    end
    #take mean accross samples
    L_NM = dropdims(mean(L_NMS, dims=3), dims=3)
    print(size(L_NM))
    #take logarithms for all heldout points
    L_NM[L_NM .!= 0] .= log.(L_NM[L_NM .!= 0])
    return exp(sum(L_NM)/I)
end