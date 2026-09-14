# Direct block-level comparison: for a FIXED conditioning state, does the rewritten
# tau_m / beta_n draw have the same mean and covariance as the original code's?
#
#   julia -t 1 --project=. revision2_experiments/covid/validate_blocks.jl
#
# This isolates the Polya-Gamma Gaussian blocks from everything else in the sweep.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "helper/MatrixMF.jl"))
using Random, Printf, Statistics, LinearAlgebra, Distributions

Random.seed!(5)
N, M, Q, Dmax = 40, 7, 4, 5
sigma2_tau, sigma2_beta = 1.7, 2.3
tauc, taud, start_tau = 1.0, 0.0, 0.0
c0 = (Dmax - 1)/4

Beta_NQ = randn(N, Q)
Tau_QM  = randn(Q, M)
W_NM    = rand(N, M) .+ 0.2
D_NM    = 2 .* rand(0:((Dmax-1)÷2), N, M) .+ 1
Kap_NM  = [(D_NM[n,m] - 1)/2 - c0 for n in 1:N, m in 1:M]

# ---------------- original tau draw for one column m (verbatim structure) -------------
function tau_draw_orig(m, Beta_NQ, Tau_QM, W_NM, D_NM, sigma2_tau, tauc, taud, start_tau, Q, M, Dmax)
    BW = Beta_NQ .* W_NM[:,m]
    V = m == M ? inv(BW' * Beta_NQ + (1/sigma2_tau)*I) :
                 inv(BW' * Beta_NQ + (1/sigma2_tau + tauc^2 / sigma2_tau)*I)
    V = .5(V + V')
    k = (D_NM[:,m] .- 1)/2 .- (Dmax - 1)/4
    if m == 1
        mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*I*(taud .+ tauc*fill(start_tau,Q)) + (tauc/sigma2_tau)*I*(Tau_QM[:,m+1] .- taud)))
    elseif m == M
        mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*I*(taud .+ tauc*Tau_QM[:,m-1])))
    else
        mvec = Float64.(V*(Beta_NQ' * k + (1/sigma2_tau)*I*(taud .+ tauc*Tau_QM[:,m-1]) + (tauc/sigma2_tau)*I*(Tau_QM[:,m+1] .- taud)))
    end
    return rand(MvNormal(mvec, V)), mvec, V
end

# ---------------- rewritten tau draw for one column m ---------------------------------
function tau_draw_new(m, Beta_NQ, Tau_QM, W_NM, Kap_NM, sigma2_tau, tauc, taud, start_tau, Q, M)
    lam_own = 1/sigma2_tau; lam_next = tauc^2 / sigma2_tau
    BW = Matrix{Float64}(undef, size(Beta_NQ)...)
    @inbounds for q in 1:Q, n in 1:size(Beta_NQ,1)
        BW[n,q] = Beta_NQ[n,q] * W_NM[n,m]
    end
    A = transpose(BW) * Beta_NQ
    lam = m == M ? lam_own : lam_own + lam_next
    for q in 1:Q; A[q,q] += lam; end
    Fc = cholesky!(Symmetric(A))
    rhs = transpose(Beta_NQ) * Kap_NM[:,m]
    for q in 1:Q
        tprev = m == 1 ? start_tau : Tau_QM[q,m-1]
        rhs[q] += lam_own * (taud + tauc*tprev)
        if m < M
            rhs[q] += (tauc/sigma2_tau) * (Tau_QM[q,m+1] - taud)
        end
    end
    ldiv!(Fc, rhs)
    z = randn(Q); ldiv!(Fc.U, z)
    return rhs .+ z, copy(rhs)
end

println("="^72)
println("tau_m block: mean and covariance, original vs rewrite")
println("="^72)
nrep = 150_000
for m in (1, 3, M)
    So = zeros(Q, nrep); Sn = zeros(Q, nrep)
    Random.seed!(100+m)
    for i in 1:nrep
        So[:,i], _, _ = tau_draw_orig(m, Beta_NQ, Tau_QM, W_NM, D_NM, sigma2_tau, tauc, taud, start_tau, Q, M, Dmax)
    end
    Random.seed!(200+m)
    for i in 1:nrep
        Sn[:,i], _ = tau_draw_new(m, Beta_NQ, Tau_QM, W_NM, Kap_NM, sigma2_tau, tauc, taud, start_tau, Q, M)
    end
    mo, mn = vec(mean(So,dims=2)), vec(mean(Sn,dims=2))
    Co, Cn = cov(So,dims=2), cov(Sn,dims=2)
    se = sqrt.(diag(Co)./nrep .+ diag(Cn)./nrep)
    zmax = maximum(abs.(mo .- mn) ./ se)
    crel = maximum(abs.(Co .- Cn)) / maximum(abs.(Co))
    @printf("m=%d  max |dmean|/se = %6.2f   max rel dcov = %.4f   %s\n",
            m, zmax, crel, (zmax < 5 && crel < 0.03) ? "OK" : "  <-- MISMATCH")
    if !(zmax < 5 && crel < 0.03)
        println("   orig mean: ", round.(mo, digits=4))
        println("   new  mean: ", round.(mn, digits=4))
        println("   orig diag cov: ", round.(diag(Co), digits=4))
        println("   new  diag cov: ", round.(diag(Cn), digits=4))
    end
end

# ---------------- beta_n block --------------------------------------------------------
function beta_draw_orig(n, Beta_NQ, Tau_QM, W_NM, D_NM, sigma2_beta, Q, Dmax)
    W = Diagonal(W_NM[n,:])
    V = inv(Tau_QM * W * Tau_QM' + (1/sigma2_beta)*Matrix{Int}(I, Q, Q))
    V = .5(V + V')
    k = (D_NM[n,:] .- 1)/2 .- (Dmax - 1)/4
    mvec = Float64.(V*(Tau_QM * k))
    return rand(MvNormal(mvec, V))
end

function beta_draw_new(n, Tau_QM, Wt_MN, Kapt_MN, sigma2_beta, Q, M)
    TW = Matrix{Float64}(undef, Q, M)
    @inbounds for m in 1:M
        wnm = Wt_MN[m,n]
        for q in 1:Q; TW[q,m] = Tau_QM[q,m] * wnm; end
    end
    A = TW * transpose(Tau_QM)
    for q in 1:Q; A[q,q] += 1/sigma2_beta; end
    Fc = cholesky!(Symmetric(A))
    rhs = Tau_QM * Kapt_MN[:,n]
    ldiv!(Fc, rhs)
    z = randn(Q); ldiv!(Fc.U, z)
    return rhs .+ z
end

Wt_MN = permutedims(W_NM)
Kapt_MN = permutedims(Kap_NM)

println()
println("="^72)
println("beta_n block: mean and covariance, original vs rewrite")
println("="^72)
for n in (1, 17, N)
    So = zeros(Q, nrep); Sn = zeros(Q, nrep)
    Random.seed!(300+n)
    for i in 1:nrep; So[:,i] = beta_draw_orig(n, Beta_NQ, Tau_QM, W_NM, D_NM, sigma2_beta, Q, Dmax); end
    Random.seed!(400+n)
    for i in 1:nrep; Sn[:,i] = beta_draw_new(n, Tau_QM, Wt_MN, Kapt_MN, sigma2_beta, Q, M); end
    mo, mn = vec(mean(So,dims=2)), vec(mean(Sn,dims=2))
    Co, Cn = cov(So,dims=2), cov(Sn,dims=2)
    se = sqrt.(diag(Co)./nrep .+ diag(Cn)./nrep)
    zmax = maximum(abs.(mo .- mn) ./ se)
    crel = maximum(abs.(Co .- Cn)) / maximum(abs.(Co))
    @printf("n=%d  max |dmean|/se = %6.2f   max rel dcov = %.4f   %s\n",
            n, zmax, crel, (zmax < 5 && crel < 0.03) ? "OK" : "  <-- MISMATCH")
    if !(zmax < 5 && crel < 0.03)
        println("   orig mean: ", round.(mo, digits=4))
        println("   new  mean: ", round.(mn, digits=4))
        println("   orig diag cov: ", round.(diag(Co), digits=4))
        println("   new  diag cov: ", round.(diag(Cn), digits=4))
    end
end
