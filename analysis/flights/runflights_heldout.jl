println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
type = parse(Int, ARGS[3])
D = parse(Int, ARGS[4])
g = parse(Int, ARGS[5])
T = 99 #because running on full
#thread = Bool(parse(Int, ARGS[1]))

datafile = "/home/jlederman/DiscreteOrderStatistics/data/flights_data/flights_final.csv"
df = CSV.read(datafile, DataFrame)
df = select(df, Not(1))
Y_NM = convert(Matrix{Int}, Matrix(select(df, 1)))
dist_NM = convert(Matrix{Int}, Matrix(select(df, 2)))
I_N3 = convert(Matrix{Int}, Matrix(select(df, 3:4)))

data = Dict("Y_NM"=>Y_NM)
info = Dict("I_N3"=>I_N3, "dist_NM"=>dist_NM)

home_N = info["I_N3"][:,1]
away_N = info["I_N3"][:,2]
dist_NM = info["dist_NM"]
N = length(home_N)

routes_R4 = Vector{Any}()
for t1 in 1:T
    for t2 in 1:T
        bitvector = home_N .== t1 .&& away_N .== t2
        if sum(bitvector) != 0
            indices = findall(bitvector)
            distances = dist_NM[indices, 1]
            @assert all(x -> x == distances[1], distances) "Not all distances are equal"
            push!(routes_R4, [t1,t2,indices, distances[1],Y_NM[indices,1],length(indices)])
        end
    end
end
routes_R4 = Matrix(hcat(routes_R4...)')
#println(routes_R4)
routes_N = zeros(Int, N)
for n in 1:N 
    t1 = home_N[n]
    t2 = away_N[n]
    routes_N[n] = findfirst(r -> r[1] == t1 && r[2] == t2, eachrow(routes_R4))
end
I_N3 = hcat(I_N3, routes_N)


R = size(routes_R4)[1]

N = size(Y_NM)[1]
M = size(Y_NM)[2]
a =1
b =.01
# c =10
# d=1
alpha = beta = 1 #not used
tau2 = 50^2
gforwards = [x -> x, sqrt]
gbakwards = [x -> x, x -> x^2]

Nsamples = 500
nbunrin = 1000
nthin = 1


Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .05

#need to format things such that the model doesn't see the test data at all
Ntest = sum(mask_NM)
Ntrain = N - Ntest
routes_R2 = routes_R4[:,3:6]
datatrain = Dict("Y_NM"=>reshape(data["Y_NM"][mask_NM .== 0],:,1), "routes_R2"=>routes_R2, "routes_N"=>routes_N[mask_NM[:,1] .== 0])
#datatest = Dict("Y_NM"=>reshape(data["Y_NM"][mask_NM .== 1],:,1), "routes_R2"=>routes_R2, "routes_N"=>routes_N[mask_NM[:,1] .== 1])
routes_R2train = copy(routes_R2)
for r in 1:R
    routes_R2train[r,1] = findall(datatrain["routes_N"] .== r)
    routes_R2train[r,2] = length(findall(datatrain["routes_N"] .== r))
end
datatrain = Dict("Y_NM"=>reshape(data["Y_NM"][mask_NM .== 0],:,1), "routes_R2"=>routes_R2train, "routes_N"=>routes_N[mask_NM[:,1] .== 0])
info = Dict("routes_R2"=>routes_R2, "routes_N"=>routes_N[mask_NM[:,1] .== 0])
if type == 1
    Dmax = 9
    include("/home/jlederman/DiscreteOrderStatistics/models/flights_final/flights.jl")
    model = flights(Ntrain, M, R, Dmax, a, b, alpha, beta)
    if D == 0
        t = @elapsed samples = fit(model, datatrain, nsamples = Nsamples, nburnin=nbunrin, nthin=nthin, mask=nothing, info=info, initseed=chainSeed,
        verbose =true)
        samplesnew = [Dict("U_R"=> s["U_R"], "D_R" => s["D_R"]) for s in samples]
    elseif D > 0
        t = @elapsed samples = fit(model, datatrain, nsamples = Nsamples, nburnin=nbunrin, nthin=nthin, mask=nothing, info=info, initseed=chainSeed,
        constantinit=Dict("D_R"=>fill(D,R)),skipupdatealways=["D_R"], verbose =true)
        samplesnew = [Dict("U_R"=> s["U_R"], "D_R" => s["D_R"]) for s in samples]
    end
elseif type == 2
    gforward = gforwards[g]
    gbakward = gbakwards[g]
    include("/home/jlederman/DiscreteOrderStatistics/models/flights_final/flights_STAR.jl")
    model = flights_STAR(Ntrain, M, R, alpha, beta, tau2, gforward, gbakward)
    t = @elapsed samples = fit(model, datatrain, nsamples = Nsamples, nburnin=nbunrin, nthin=nthin, mask=nothing, info=info, initseed=chainSeed,
    verbose =true)
    samplesnew = [Dict("U_R"=> s["U_R"], "sigma2_R" => s["sigma2_R"]) for s in samples]
end

params = [maskSeed,chainSeed,type,D,g]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/flights/"
if type == 1
    save(folder*"revisionsamples/MedPoissonD$(D)mask$(maskSeed)chain$(chainSeed).jld", "params", params, "samples", samplesnew, "mask", mask_NM, "time", t)
elseif type == 2
    save(folder*"revisionsamples/STARg$(g)mask$(maskSeed)chain$(chainSeed).jld", "params", params, "samples", samplesnew, "mask", mask_NM, "time", t)
end