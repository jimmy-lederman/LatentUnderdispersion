println("hello World")
# using Pkg
# Pkg.add("LinearAlgebra")
using Distributions
x = [1,2,3]
y= [2,4,6]
@assert x == y/2
N = 5
M = 10
A_NM = rand(N,M)
B_M = 1:10

mu = 1
Y = 2
dist = Poisson(mu)
mat_NM = rand(dist, N, M)

dist = Truncated(Poisson(mu), 0, Y)
mat_NM = rand(dist, N, M)


K = 3
D = 4

dist = Gamma(1,1)
U_NK = rand(dist,N,K)
V_KM = rand(dist,K,M)
mu_NM = U_NK*V_KM
println(mu_NM)
A_NM = rand.(Poisson.(mu_NM))
println(A_NM)

mu_NMD = repeat(mu_NM, inner = (1, 1, D))
A_NMD = rand.(Poisson.(mu_NMD))
println(A_NMD)
println(size(A_NMD))

max_NM = dropdims(maximum(A_NMD, dims =3), dims=3)
println(max_NM)
println(size(max_NM))


mu_NM = U_NK*V_KM
A_NM = rand.(OrderStatistic.(Poisson.(mu_NM), D, D))
println(A_NM)

println(mu_NM)
println(pdf.(OrderStatistic.(Poisson.(mu_NM), 1, 1), A_NM))