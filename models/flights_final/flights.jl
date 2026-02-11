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
    route = info["routes_N"][row]
    mu = state["U_R"][route]
    D = state["D_R"][route]
    
    if D == 1
        return logpdf(Poisson(mu), Y)
    else
        #return logpmfOrderStatPoisson(Y,mu,D,div(D,2)+1)
        return logpdf(OrderStatistic(Poisson(mu),D,div(D,2)+1), Y)
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

    state = Dict("U_R" => U_R, "p"=>p, "D_R"=>D_R,
                "routes_R2"=>info["routes_R2"],
                "routes_N"=>info["routes_N"]
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

function backward_sample(model::flights, data, state, mask=nothing; skipupdatealways=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    routes_N = data["routes_N"]
    routes_R2 = data["routes_R2"]
    U_R = copy(state["U_R"])
    D_R = copy(state["D_R"])
    p = copy(state["p"])



    @assert model.M == 1

    #3: the distance*cluster mean

    #Z1_NM = zeros(Int, model.N,1)

    nt = Threads.nthreads()
    Z_R_nt = [zeros(Int, model.R) for _ in 1:nt]
    #unfortunately, to impute the held out data points and
    #sample poissons from maximum, we must loop over N
    @views @threads for n in 1:model.N  
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
        z = sampleSumGivenOrderStatistic(Y_NM[n, 1], D, div(D,2)+1, Poisson(mu))
        Z_R_nt[tid][r] += z
        #end
    end
    Z_R  = sum(Z_R_nt)  


    #now that we have latent Poissons, additivity allows us to
    #loop over R (R <<< N)
    @views for r in 1:model.R
        #indices = routes_R4[r,3]
        # println(indices)
        numflights =  routes_R2[r,2]
        post_shape = model.a + Z_R[r]
        post_rate = model.b + D_R[r]*numflights
        U_R[r] = rand(Gamma(post_shape, 1 / post_rate))
    end

    if isnothing(skipupdatealways) || !("D_R" in skipupdatealways)
        # prior part (no threading needed)
        logprobs_prior = [
            logbinomial(Int((model.Dmax - 1) ÷ 2), Int((d - 1) ÷ 2)) +
            (d - 1) * log(p) / 2 +
            (model.Dmax - d) * log(1 - p) / 2
            for d in 1:2:model.Dmax
        ]
        @views @threads for r in 1:model.R
            Ys = routes_R2[r, 1]
            mu = U_R[r]

            # initialize logprobs from prior
            logprobs = copy(logprobs_prior)

            # main likelihood accumulation
            for d in 1:2:model.Dmax
                j = (d ÷ 2) + 1
                if d == 1
                    dist = Poisson(mu)
                else
                    dist = OrderStatistic(Poisson(mu),d,j)
                end
                 
                logprobs[j] += sum(logpdf.(dist,Ys))
                # if (d == 1 || d== 3) && r == 474

                #      println(logpdf.(dist,Ys))


                # end

            end
            # if r == 474
            #     println(Ys)
            #     println(mu)
            #     println(logprobs)
            # end
            # Gumbel-max trick
            D_R[r] = 2 * argmax(rand(Gumbel(0, 1), length(logprobs)) .+ logprobs) - 1
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