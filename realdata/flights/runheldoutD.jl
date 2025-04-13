println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

D = parse(Int, ARGS[1])
maskSeed = parse(Int, ARGS[2])
chainSeed = parse(Int, ARGS[3])
T = 99 #because running on full
#thread = Bool(parse(Int, ARGS[1]))

datafile = "/home/jlederman/DiscreteOrderStatistics/data/airtime.csv"
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
            push!(routes_R4, [t1,t2,indices, distances[1]])
        end
    end
end
Matrix(routes_R4')
routes_R4 = Matrix(hcat(routes_R4...)')
#println(routes_R4)
routes_N = zeros(Int, N)
for n in 1:N 
    t1 = home_N[n]
    t2 = away_N[n]
    routes_N[n] = findfirst(r -> r[1] == t1 && r[2] == t2, eachrow(routes_R4))
end
I_N3 = hcat(I_N3, routes_N)

info = Dict("routes_R4"=>routes_R4, "I_N3"=>I_N3, "dist_N"=>dist_NM)
R = size(routes_R4)[1]

N = size(Y_NM)[1]
M = size(Y_NM)[2]
a =20
b =100
c =10
d=1
alpha = beta = 1 #not used

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .05

if D == 0
    Dmax = 9
    include("/home/jlederman/DiscreteOrderStatistics/models/flights/flights3.jl")
    model = flights(N, M, T, R, Dmax, a, b, c, d, alpha, beta)
    @time samples = fit(model, data, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM, info=info,initseed=chainSeed)
    samplesnew = [Dict("U_R"=> s["U_R"], "D_R"=>s["D_R"], "A_T"=>s["A_T"], "B_T"=>s["B_T"], "p"=>s["p"]) for s in samples]
else
    include("/home/jlederman/DiscreteOrderStatistics/models/flights/flights3base.jl")
    model = flightsbase(N, M, T, R, D, a, b, c, d, alpha, beta)
    @time samples = fit(model, data, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM, info=info,initseed=chainSeed)
    samplesnew = [Dict("U_R"=> s["U_R"],"A_T"=>s["A_T"], "B_T"=>s["B_T"]) for s in samples]
end

#inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, info=info, verbose=false)
params = [D,maskSeed,chainSeed]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/flights/"
save(folder*"heldoutsamplesD/sampleD$(D)seedMask$(maskSeed)seedChain$(chainSeed).jld", "params", params, "samples", samplesnew, "I_N3",  samples[1]["I_N3"], "dist_N", samples[1]["dist_N"], "routes_R4", samples[1]["routes_R4"])
