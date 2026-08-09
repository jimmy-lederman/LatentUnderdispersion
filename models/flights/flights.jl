include("../../helper/MatrixMF.jl")
include("../../helper/OrderStatsSampling.jl")
include("../../helper/PoissonOrderPMF.jl")
using Distributions
using Base.Threads

struct flights <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    Dmax::Int64
    a::Float64
    b::Float64
    alpha::Float64
    beta::Float64
end

function evalulateLogLikelihood(model::flights, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    @assert size(data["Y_NM"])[2] == 1
    route = data["routes_N"][row]
    mu = state["U_R"][route]
    D = state["D_R"][route]
    
    if D == 1
        x = logpdf(Poisson(mu), Y)

        if !isfinite(x)
            println(x)
            println(mu)
            println(route)
            @assert 1 == 2
        end
        return x
    else
        return logpmfOrderStatPoisson(Y,mu,D,div(D,2)+1)
        #return logpdf(OrderStatistic(Poisson(mu),D,div(D,2)+1), Y)
    end
end



function sample_likelihood(model::flights, mu,D,n=1)
    if D == 1
        if n == 1
            return rand(Poisson(mu))
        else
            return rand(Poisson(mu),n)
        end
    else
        if n == 1
            j = div(D,2) + 1
            return rand(OrderStatistic(Poisson(mu), D, j))
        else
            j = div(D,2) + 1
            return rand(OrderStatistic(Poisson(mu), D, j),n)
        end
    end
end

function sample_prior(model::flights, info=nothing, constantint=nothing)
    p = nothing
    U_R = rand(Gamma(model.a, 1/model.b), model.R)
    # p = rand(Beta(model.alpha,model.beta)) 
    p = .5
    @assert mod(model.Dmax, 2) == 1
    D_R = 2 * rand(Binomial((model.Dmax - 1)/2, p), model.R) .+ 1

    state = Dict("U_R" => U_R, "p"=>p, "D_R"=>D_R
                )
    return state
end

function forward_sample(model::flights; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    U_R = state["U_R"]

    D_R =state["D_R"]
    routes_N = state["routes_N"]

    @assert model.M == 1
    Y_NM = zeros(Int, model.N,model.M)
    for n in 1:model.N
        route = routes_N[n]
        mu = U_R[route]
        D = D_R[route]
        Y_NM[n,1] = sample_likelihood(model,mu,D)
    end
    data = Dict("Y_NM" => Y_NM)
    return data, state
end

logbinomial(n::Integer, k::Integer) = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

#the D update scores every candidate D against every flight on a route, but the parent
#Poisson cdf/pmf do not depend on D and repeated Y values on a route contribute the same
#term. So we tabulate the distinct Y values per route once and weight by multiplicity.
#When mask === nothing the training Y are fixed across sweeps, so this is built once.
function route_ycounts(Y_NM, route_idx::Vector{Vector{Int}})
    R = length(route_idx)
    uY = Vector{Vector{Int}}(undef, R)
    cY = Vector{Vector{Int}}(undef, R)
    for r in 1:R
        counts = Dict{Int,Int}()
        for n in route_idx[r]
            y = Y_NM[n, 1]
            counts[y] = get(counts, y, 0) + 1
        end
        ks = sort!(collect(keys(counts)))
        uY[r] = ks
        cY[r] = [counts[k] for k in ks]
    end
    return uY, cY
end

function backward_sample(model::flights, data, state, mask=nothing; skipupdatealways=nothing, skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    routes_N = data["routes_N"]
    route_idx = data["route_idx"]::Vector{Vector{Int}}
    route_n = data["route_n"]::Vector{Int}
    U_R = copy(state["U_R"])
    D_R = Int.(copy(state["D_R"]))
    p = copy(state["p"])


    @assert model.M == 1

    #3: the distance*cluster mean

    #Z1_NM = zeros(Int, model.N,1)

    nt = Threads.nthreads()
    Z_R_nt = [zeros(Int, model.R) for _ in 1:nt]
    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    #NOTE: :static is required -- indexing Z_R_nt by threadid() is only safe when tasks
    #cannot migrate between threads, which the default :dynamic schedule does not guarantee.
    @views @threads :static for n in 1:model.N
        tid = Threads.threadid()
        r =  routes_N[n]
        mu = U_R[r]
        D = D_R[r]

        if !isnothing(mask) && mask[n,1] == 1
            if D > 1
                j = div(D,2) + 1
                Y_NM[n,1] = rand(OrderStatistic(Poisson(mu), D, j))
            else
                Y_NM[n,1] = rand(Poisson(mu))
            end
        end
        #if Y_NM[n, 1] > 0
        z = 0
        try
            z = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, div(D,2)+1, Poisson(mu))
        catch ex
            println("sampleSumGivenOrderStatistic failed: n=$n route=$r Y=$(Y_NM[n,1]) D=$D mu=$mu")
            rethrow(ex)
        end

        Z_R_nt[tid][r] += z
        #end
    end
    Z_R  = sum(Z_R_nt)


    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    @views for r in 1:model.R
        numflights = route_n[r]
        post_shape = model.a + Z_R[r]
        post_rate = model.b + D_R[r]*numflights
        U_R[r] = rand(Gamma(post_shape, 1 / post_rate))
    end
    go = true
    if !isnothing(skipupdatealways)
        if "D_R" in skipupdatealways 
            go = false
        end
    end
    if !isnothing(skipupdate)
        if "D_R" in skipupdate 
            go = false
        end
    end

    if go

        # prior part (no threading needed)
        logprobs_prior = [
            logbinomial(Int((model.Dmax - 1) ÷ 2), Int((d - 1) ÷ 2)) +
            (d - 1) * log(p) / 2 +
            (model.Dmax - d) * log(1 - p) / 2
            for d in 1:2:model.Dmax
        ]

        #distinct Y values per route, with multiplicities. Cached when nothing is imputed.
        tabs = isnothing(mask) ?
            get!(() -> route_ycounts(Y_NM, route_idx), data, "route_ycounts") :
            route_ycounts(Y_NM, route_idx)
        uY = tabs[1]::Vector{Vector{Int}}
        cY = tabs[2]::Vector{Vector{Int}}

        cand = collect(1:2:model.Dmax)
        ncand = length(cand)
        #coefficients for the cancellation-free order statistic pmf, one table per candidate
        #D, built once per sweep rather than once per route
        tabs = [orderstat_grid_coefs(d, (d ÷ 2) + 1) for d in cand]

        @views @threads :static for r in 1:model.R
            mu = U_R[r]
            dist = Poisson(mu)
            logprobs = copy(logprobs_prior)
            pA = Vector{Float64}(undef, model.Dmax + 1)
            pE = Vector{Float64}(undef, model.Dmax + 1)
            pB = Vector{Float64}(undef, model.Dmax + 1)
            uYr = uY[r]
            cYr = cY[r]

            for i in eachindex(uYr)
                y = uYr[i]
                c = cYr[i]
                #these do not depend on the candidate D, so evaluate once and reuse
                F = cdf(dist, y)
                f = pdf(dist, y)
                lf = logpdf(dist, y)
                for ci in 1:ncand
                    d = cand[ci]
                    if d == 1
                        logprobs[ci] += c * lf
                    else
                        j = (d ÷ 2) + 1
                        v = logpmf_orderstat_grid(F, f, d, j, tabs[ci], pA, pE, pB)
                        #grid underflowed; fall back to the BigFloat-guarded pmf
                        isnan(v) && (v = logpmfOrderStatPoisson(y, mu, d, j))
                        logprobs[ci] += c * v
                    end
                end
            end

            # Gumbel-max trick
            D_R[r] = 2 * argmax(rand(Gumbel(0, 1), ncand) .+ logprobs) - 1
        end
        p = rand(Beta(model.alpha + (sum(D_R)- model.R)/2, model.beta + (model.Dmax*model.R - sum(D_R))/2))

    end


    state = Dict("U_R" => U_R,
                "p"=>p,
                "D_R"=>D_R)
    return data, state
end

              # for f in 1:length(indices)
                #     tid = Threads.threadid()
                #     Y = Y_NM[indices[f], 1]
                #     # acc[tid] += logpmfOrderStatPoisson(
                #     #     Y, mu, d, didx
                #     # )
                #     acc += logpmfOrderStatPoisson(
                #         Y, mu, d, didx
                #     )
                # end