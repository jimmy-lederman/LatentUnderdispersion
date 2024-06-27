import Pkg; 
# Pkg.add("PyPlot")
using PyPlot # plotting for Julia based on matplotlib.pyplot
# Pkg.add("Distributions")
using Distributions # probability distributions and associated functions
# Pkg.add("Pandas")
using Pandas # a front-end to work with Python’s Pandas.
# Pkg.add("PyCall")
using PyCall # call Python functions.

# Pkg.add("ProgressBars")
using ProgressBars

# Pkg.add("StatsFuns")
using StatsFuns: logsumexp

x = [1, 2, 3]  # column vector (1 x 3)
y = [1  2  3]  # row vector (3 x 1)

# assert that x transpose equals y
@assert x' == y

# create a 3 x 3 matrix
A = [1 2 3; 
     4 5 6; 
     7 8 9]

# sample a (D x K) matrix of iid gamma random variables with shape 0.01 and scale 100 
K = 10
D = 100
V = 200
shape = 0.1
scale = 100

# Generate under the Poisson matrix factorization model
true_A_DK = rand(Gamma(shape, scale), (D, K))
true_B_KV = rand(Gamma(shape, scale), (K, V))
Y_DV = rand.(Poisson.(true_A_DK * true_B_KV))

# print the number of non-zeros
println("Number of non-zeros: ", sum(Y_DV .> 0))

# Run Gibbs sampling
A_DK = rand(Gamma(shape, scale), (D, K))
B_KV = rand(Gamma(shape, scale), (K, V))
Y_DVK = zeros(D, V, K)
P_K = zeros(K)


n_burnin = 250
n_samples = 1000
S = n_burnin + n_samples

for s in ProgressBar(1:S)
    # Loop over the non-zeros in Y_DV and allocate
    for d in 1:D
        for v in 1:V
            if Y_DV[d, v] > 0
                P_K[:] = A_DK[d, :] .* B_KV[:, v]
                P_K[:] = P_K / sum(P_K)
                Y_DVK[d, v, :] = rand(Multinomial(Y_DV[d, v], P_K))
            end
        end
    end

    # how do I vectorize this? should I?
    for d in 1:D
        for k in 1:K
            post_shape = shape + sum(Y_DVK[d, :, k])
            post_rate = 1/scale + sum(B_KV[k, :])
            A_DK[d, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for v in 1:V
        for k in 1:K
            post_shape = shape + sum(Y_DVK[:, v, k])
            post_rate = 1/scale + sum(A_DK[:, k])
            B_KV[k, v] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
end
