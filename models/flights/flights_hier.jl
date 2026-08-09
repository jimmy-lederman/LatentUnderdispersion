# Hierarchical-mu variant of the flights MedPoisson model (referee comment 2(a):
# "did the authors consider a hierarchical model for mu_k across routes?").
#
#   Y_i      ~ MedPois(mu_route[i], D_route[i])
#   mu_k | b ~ Gamma(a, rate b)      a FIXED at 1, exactly as in the flat model
#   b        ~ Gamma(a0, rate b0)    conjugate -- the only change
#
# The flat model is the special case b = 0.01 held fixed; here the population rate is
# learned from all R routes instead, so the routes share information through it. The shape
# a is deliberately NOT given a prior: keeping it at its original value makes this a
# one-line modification with no new tuning choices and no non-conjugate step.
#
# Everything except the mu/b block is identical to models/flights/flights.jl, which this
# file includes (and whose route_ycounts / logbinomial it reuses).

include("flights.jl")

struct flights_hier <: MatrixMF
    N::Int64
    M::Int64
    R::Int64
    Dmax::Int64
    a::Float64       # population shape, FIXED (1.0, as in the flat model)
    a0::Float64      # hyperprior shape for the population rate b
    b0::Float64      # hyperprior rate  for the population rate b
    alpha::Float64
    beta::Float64
end

function evalulateLogLikelihood(model::flights_hier, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    route = data["routes_N"][row]
    mu = state["U_R"][route]
    D = state["D_R"][route]
    return D == 1 ? logpdf(Poisson(mu), Y) : logpmfOrderStatPoisson(Y, mu, D, div(D,2)+1)
end

function sample_likelihood(model::flights_hier, mu, D, n=1)
    if D == 1
        return n == 1 ? rand(Poisson(mu)) : rand(Poisson(mu), n)
    else
        j = div(D,2) + 1
        return n == 1 ? rand(OrderStatistic(Poisson(mu), D, j)) :
                        rand(OrderStatistic(Poisson(mu), D, j), n)
    end
end

function sample_prior(model::flights_hier, info=nothing, constantint=nothing)
    bmu = rand(Gamma(model.a0, 1/model.b0))
    U_R = rand(Gamma(model.a, 1/bmu), model.R)
    p = rand(Beta(model.alpha, model.beta))   #rho, drawn from its prior (was hardcoded .5)
    @assert mod(model.Dmax, 2) == 1
    D_R = 2 * rand(Binomial((model.Dmax - 1)/2, p), model.R) .+ 1
    return Dict("U_R"=>U_R, "p"=>p, "D_R"=>D_R, "bmu"=>bmu)
end

function forward_sample(model::flights_hier; state=nothing, info=nothing)
    if isnothing(state); state = sample_prior(model, info); end
    U_R = state["U_R"]; D_R = state["D_R"]; routes_N = state["routes_N"]
    @assert model.M == 1
    Y_NM = zeros(Int, model.N, model.M)
    for n in 1:model.N
        Y_NM[n,1] = sample_likelihood(model, U_R[routes_N[n]], D_R[routes_N[n]])
    end
    return Dict("Y_NM"=>Y_NM), state
end

function backward_sample(model::flights_hier, data, state, mask=nothing; skipupdatealways=nothing, skipupdate=nothing)
    Y_NM = copy(data["Y_NM"])
    routes_N = data["routes_N"]
    route_idx = data["route_idx"]::Vector{Vector{Int}}
    route_n = data["route_n"]::Vector{Int}
    U_R = copy(state["U_R"])
    D_R = Int.(copy(state["D_R"]))
    p = copy(state["p"])
    bmu = copy(state["bmu"])
    @assert model.M == 1

    # ---- latent Poisson sums (identical to the flat model) ----
    nt = Threads.nthreads()
    Z_R_nt = [zeros(Int, model.R) for _ in 1:nt]
    @views @threads :static for n in 1:model.N
        tid = Threads.threadid()
        r = routes_N[n]
        mu = U_R[r]; D = D_R[r]
        if !isnothing(mask) && mask[n,1] == 1
            Y_NM[n,1] = D > 1 ? rand(OrderStatistic(Poisson(mu), D, div(D,2)+1)) : rand(Poisson(mu))
        end
        Z_R_nt[tid][r] += sampleSumGivenOrderStatistic(Y_NM[n,1], D, div(D,2)+1, Poisson(mu))
    end
    Z_R = sum(Z_R_nt)

    # ---- mu_k | b : same conjugacy as the flat model, with the LEARNED rate b ----
    @views for r in 1:model.R
        U_R[r] = rand(Gamma(model.a + Z_R[r], 1 / (bmu + D_R[r]*route_n[r])))
    end

    # ---- population rate b | mu : conjugate. This is the whole modification. ----
    bmu = rand(Gamma(model.a0 + model.R*model.a, 1 / (model.b0 + sum(U_R))))

    # ---- D_k update (identical to the flat model) ----
    go = true
    if !isnothing(skipupdatealways) && "D_R" in skipupdatealways; go = false; end
    if !isnothing(skipupdate) && "D_R" in skipupdate; go = false; end

    if go
        logprobs_prior = [
            logbinomial(Int((model.Dmax - 1) ÷ 2), Int((d - 1) ÷ 2)) +
            (d - 1) * log(p) / 2 + (model.Dmax - d) * log(1 - p) / 2
            for d in 1:2:model.Dmax
        ]
        tabs_y = isnothing(mask) ?
            get!(() -> route_ycounts(Y_NM, route_idx), data, "route_ycounts") :
            route_ycounts(Y_NM, route_idx)
        uY = tabs_y[1]::Vector{Vector{Int}}
        cY = tabs_y[2]::Vector{Vector{Int}}
        cand = collect(1:2:model.Dmax); ncand = length(cand)
        tabs = [orderstat_grid_coefs(d, (d ÷ 2) + 1) for d in cand]

        @views @threads :static for r in 1:model.R
            mu = U_R[r]; dist = Poisson(mu)
            logprobs = copy(logprobs_prior)
            pA = Vector{Float64}(undef, model.Dmax + 1)
            pE = Vector{Float64}(undef, model.Dmax + 1)
            pB = Vector{Float64}(undef, model.Dmax + 1)
            uYr = uY[r]; cYr = cY[r]
            for i in eachindex(uYr)
                y = uYr[i]; c = cYr[i]
                F = cdf(dist, y); f = pdf(dist, y); lf = logpdf(dist, y)
                for ci in 1:ncand
                    d = cand[ci]
                    if d == 1
                        logprobs[ci] += c * lf
                    else
                        j = (d ÷ 2) + 1
                        v = logpmf_orderstat_grid(F, f, d, j, tabs[ci], pA, pE, pB)
                        isnan(v) && (v = logpmfOrderStatPoisson(y, mu, d, j))
                        logprobs[ci] += c * v
                    end
                end
            end
            D_R[r] = 2 * argmax(rand(Gumbel(0, 1), ncand) .+ logprobs) - 1
        end
        p = rand(Beta(model.alpha + (sum(D_R)- model.R)/2, model.beta + (model.Dmax*model.R - sum(D_R))/2))
    end

    return data, Dict("U_R"=>U_R, "p"=>p, "D_R"=>D_R, "bmu"=>bmu)
end
