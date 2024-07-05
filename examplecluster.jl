include("poissonMF")
using Random


#what should ARGS be?
#1 -> data file
#2 -> mask seed
#3 -> K
#4 -> D
#5 -> output file

N = M = 10
K = 2
a = b = c = d = 1
maskSeed = 1
model = PoissonMF(N, M, K, a, b, c, d)
data, ~ = forward_sample(model)
Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .2
samples = fit(model, data, mask=mask_NM)

inforate = evaluateInfoRate(model, data, samples, mask_NM)
println("inforate")
