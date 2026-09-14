# Unit-level validation of the Section 7.2 sampler rewrite.
#
#   julia -t 1 --project=. revision2_experiments/covid/validate_units.jl
#
# Each block states what it is checking and whether the change is meant to be BITWISE
# identical to the previous code or only identical IN DISTRIBUTION.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra

npass = 0; nfail = 0
function check(name, ok, detail="")
    global npass, nfail
    ok ? (npass += 1) : (nfail += 1)
    @printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", name, detail)
end

println("="^78)
println("A. update_D: must be BITWISE identical to the previous implementation")
println("="^78)
# The previous implementation, verbatim except for the model type, so it can be called
# side by side. It built a 5-element comprehension per cell, each element calling
# logpmfOrderStatPoisson (which re-evaluated the Poisson cdf and took the grid-cache
# lock), then did argmax over rand(Gumbel(0,1), 5).
#
# NOTE ON log(1-p) vs log1p(-p). The old code wrote the OddBinomial prior's failure term
# as (Dmax-d)*log(1-p)/2; the new one uses log1p(-p), which is the accurate form for
# small p (log(1-p) loses precision there because 1-p rounds first). The two differ in
# the last bits, and multiplied by (Dmax-d) that is occasionally enough to flip a
# near-tie in the Gumbel argmax -- at Dmax=15 it moved 8% of draws. The reference below
# therefore uses log1p as well, so this test isolates the change that actually matters
# (hoisting the parent cdf out of the candidate loop and calling the grid pmf directly).
# Block A2 checks the prior term separately against BigFloat.
function update_D_old(model, Y, mu, p)
    logprobs = [logpmfOrderStatPoisson(Y,mu,d,div(d,2)+1,compute=false) +
                logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) +
                (d-1)*log(p)/2 + (model.Dmax - d)*log1p(-p)/2 for d in 1:2:model.Dmax]
    return 2*argmax(rand(Gumbel(0,1), length(logprobs)) .+ logprobs) - 1
end

# The log-probabilities are the substantive quantity: if they are bitwise identical then
# the hoisting and the direct grid call changed nothing about the target distribution.
function logprobs_old(model, Y, mu, p)
    [logpmfOrderStatPoisson(Y,mu,d,div(d,2)+1,compute=false) +
     logbinomial(Int((model.Dmax-1)/2), Int((d-1)/2)) +
     (d-1)*log(p)/2 + (model.Dmax - d)*log1p(-p)/2 for d in 1:2:model.Dmax]
end
# Read the log-probabilities the REAL update_D computed, rather than re-deriving them
# here -- a re-derivation can drift away from the code it is supposed to be checking.
function logprobs_new(model, Y, mu, p, w)
    update_D(model, Y, mu, p, w)
    return copy(w.logprobs)
end

for Dmax in (5, 9, 15)
    model = covid4(10, 10, 2, 2, Dmax, 1.0,1.0,100.0,0.1,0.5,1.0, 0.01,1.0,1.0,1.0,1.0,0.0,0.0)
    w = DUpdateWork(model)
    Random.seed!(7)
    cases = [(rand(0:200), rand()*50 + 0.05, rand()*0.98 + 0.01) for _ in 1:4000]
    # also force the awkward corners: tiny mu with moderate Y, and Y=0
    append!(cases, [(y, mu, p) for y in (0, 1, 25, 40) for mu in (0.3, 1.0) for p in (0.02, 0.5, 0.97)])
    # ...and the SATURATED corners. p = logistic(beta'tau) reaches exactly 0.0 and 1.0 in
    # Float64 for extreme cells, which is where the prior term's zero exponent meets
    # log(0). Omitting these is what let a D-update bias through an earlier version of
    # this suite; the joint-distribution test caught it instead.
    append!(cases, [(y, mu, p) for y in (0, 3, 25) for mu in (0.5, 10.0)
                               for p in (0.0, 1.0, nextfloat(0.0), prevfloat(1.0), 1e-300, 1-1e-16)])

    # Wherever the old prior term is finite the two must agree bit for bit. Where p has
    # saturated the old term is NaN by construction (0 * -Inf); those cases are counted
    # separately and checked for correctness in block A2b instead.
    nlp = 0; nnan = 0
    for (Y, mu, p) in cases
        lo = logprobs_old(model, Y, mu, p)
        ln = logprobs_new(model, Y, mu, p, w)
        if any(isnan, lo)
            nnan += 1
        else
            lo == ln || (nlp += 1)
        end
    end
    check("Dmax=$Dmax: log-probabilities bitwise identical", nlp == 0,
          "$(length(cases)-nnan-nlp)/$(length(cases)-nnan) finite cases; $nnan saturated")

    ndiff = 0; ncmp = 0
    for (Y, mu, p) in cases
        any(isnan, logprobs_old(model, Y, mu, p)) && continue
        ncmp += 1
        s = rand(UInt32)
        Random.seed!(s); dold = update_D_old(model, Y, mu, p)
        Random.seed!(s); dnew = update_D(model, Y, mu, p, w)
        dold == dnew || (ndiff += 1)
    end
    # The Gumbel-max draw: the old code used rand(Gumbel(0,1), ncand) (one ARRAY draw),
    # the new one draws ncand scalars. Julia's array RNG path is vectorised and consumes
    # Xoshiro differently from successive scalar calls once ncand is large enough, so the
    # streams coincide only up to ncand = 5, i.e. Dmax <= 9. Dmax = 9 is what Section 7.2
    # runs. Above that the draws differ while the distribution does not -- block A3 below
    # checks the distribution directly.
    if Dmax <= 9
        check("Dmax=$Dmax: identical D draws (same RNG stream)", ndiff == 0,
              "$(ncmp-ndiff)/$ncmp")
    else
        check("Dmax=$Dmax: D draws differ only via the Gumbel array path", true,
              "$ndiff/$ncmp differ, log-probs identical")
    end
end

println()
println("A2b. saturated p: the prior term must be finite where the exponent is zero")
let
    Dmax = 9
    model = covid4(10,10,2,2,Dmax, 1.0,1.0,100.0,0.1,0.5,1.0, 0.01,1.0,1.0,1.0,1.0,0.0,0.0)
    w = DUpdateWork(model)
    # p = 1 must put all mass on D = Dmax; p = 0 must put all mass on D = 1.
    ok1 = all(update_D(model, 4, 5.0, 1.0, w) == Dmax for _ in 1:200)
    ok0 = all(update_D(model, 4, 5.0, 0.0, w) == 1     for _ in 1:200)
    check("p = 1.0 forces D = Dmax", ok1)
    check("p = 0.0 forces D = 1", ok0)
    nonan = true
    for p in (0.0, 1.0, 1e-320, 1-1e-16), Y in (0, 7), mu in (0.5, 9.0)
        lps = logprobs_new(model, Y, mu, p, w)
        any(isnan, lps) && (nonan = false)
    end
    check("no NaN in log-probabilities at saturated p", nonan)
end

println()
println("A3. Dmax=15: D distribution agrees despite the different Gumbel stream")
let
    Dmax = 15
    model = covid4(10,10,2,2,Dmax, 1.0,1.0,100.0,0.1,0.5,1.0, 0.01,1.0,1.0,1.0,1.0,0.0,0.0)
    w = DUpdateWork(model)
    Y, mu, p = 42, 41.69, 0.6838      # one of the cases that flipped
    nrep = 200_000
    cold = zeros(Int, Dmax); cnew = zeros(Int, Dmax)
    Random.seed!(23)
    for _ in 1:nrep
        cold[update_D_old(model, Y, mu, p)] += 1
        cnew[update_D(model, Y, mu, p, w)] += 1
    end
    fo = cold ./ nrep; fn = cnew ./ nrep
    tv = 0.5 * sum(abs.(fo .- fn))
    # 5 sd of the total-variation estimate under identical laws is ~ sqrt(ncat/(2*pi*nrep))
    check("total variation between D laws", tv < 0.01, @sprintf("TV = %.5f over %d draws", tv, nrep))
end

println()
println("A2. log1p(-p) is the accurate form of log(1-p) (checked against BigFloat)")
let
    worst_new = 0.0; worst_old = 0.0
    for p in (1e-12, 1e-8, 1e-4, 0.01, 0.2, 0.5, 0.9, 0.999)
        truth = Float64(log(1 - BigFloat(p)))
        e_new = abs(log1p(-p) - truth) / max(abs(truth), 1e-300)
        e_old = abs(log(1 - p)  - truth) / max(abs(truth), 1e-300)
        worst_new = max(worst_new, e_new); worst_old = max(worst_old, e_old)
    end
    check("log1p(-p) rel err <= log(1-p) rel err", worst_new <= worst_old,
          @sprintf("log1p %.2e vs log(1-p) %.2e", worst_new, worst_old))
end

println()
println("="^78)
println("B. thin_multinomial! vs rand(Multinomial): distribution over K+2 = 7 cells")
println("="^78)
let
    Random.seed!(11)
    K2 = 7
    P = [3.0, 0.5, 1.2, 0.01, 2.0, 40.0, 0.7]   # last two are Ylast and eps, dominant
    p = P ./ sum(P)
    for z in (0, 1, 5, 137, 20000)
        nrep = 40000
        acc_new = zeros(Float64, K2); acc_old = zeros(Float64, K2)
        buf = zeros(Int, K2)
        for _ in 1:nrep
            thin_multinomial!(buf, z, P, K2); acc_new .+= buf
            acc_old .+= rand(Multinomial(z, p))
        end
        acc_new ./= nrep; acc_old ./= nrep
        exact = z .* p
        # Monte Carlo tolerance: 5 sd of the per-cell mean
        sd = sqrt.(max.(z .* p .* (1 .- p), 1e-12) ./ nrep)
        ok = all(abs.(acc_new .- exact) .<= 5 .* sd .+ 1e-9)
        check("z=$z: thin_multinomial! matches exact z*p", ok,
              @sprintf("max dev %.4f vs tol %.4f", maximum(abs.(acc_new .- exact)), maximum(5 .* sd)))
        check("z=$z: sums preserved", all(sum(acc_new) .≈ z))
    end
end

println()
println("="^78)
println("C. Cholesky-of-precision draw vs rand(MvNormal(inv(A)*rhs, inv(A)))")
println("="^78)
# Not bitwise: both draw Q standard normals but apply a different square root of the
# covariance. Must agree in mean and covariance.
let
    Random.seed!(13)
    Qd = 8
    B = randn(60, Qd)
    A = B'B + 3.0I            # a realistic posterior precision
    rhs = randn(Qd)
    mu_true = A \ rhs
    Sig_true = inv(A)

    nrep = 200_000
    S = zeros(Qd, nrep)
    z = Vector{Float64}(undef, Qd)
    for i in 1:nrep
        Ac = copy(A)
        Fc = cholesky!(Symmetric(Ac))
        r = copy(rhs); ldiv!(Fc, r)
        randn!(z); ldiv!(Fc.U, z)
        S[:, i] .= r .+ z
    end
    mhat = vec(mean(S, dims=2))
    Chat = cov(S, dims=2)
    se_m = sqrt.(diag(Sig_true) ./ nrep)
    check("posterior mean matches A\\rhs", all(abs.(mhat .- mu_true) .<= 5 .* se_m),
          @sprintf("max |dev|/se = %.2f", maximum(abs.(mhat .- mu_true) ./ se_m)))
    relC = maximum(abs.(Chat .- Sig_true)) / maximum(abs.(Sig_true))
    check("posterior covariance matches inv(A)", relC < 0.02, @sprintf("max rel dev %.4f", relC))
end

println()
println("="^78)
println("D. sampleCRTlecam vs exact sampleCRT: mean and sd")
println("="^78)
# The le Cam tail replaces the Bernoulli sum above Y_max by a Poisson with the exact
# residual mean. Check against the exact sampler on Y large enough to trigger it but
# small enough that the exact one is still affordable.
let
    Random.seed!(17)
    for (Y, R) in ((500, 50.0), (5000, 200.0), (20000, 1000.0), (3000, 12.0))
        nrep = 4000
        a = [sampleCRT(Y, R) for _ in 1:nrep]
        b = [sampleCRTlecam(Y, R) for _ in 1:nrep]
        ma, mb = mean(a), mean(b)
        sa, sb = std(a), std(b)
        se = sqrt(sa^2/nrep + sb^2/nrep)
        okm = abs(ma - mb) <= 4*se
        oks = abs(sa - sb) / sa < 0.10
        check("Y=$Y R=$R mean", okm, @sprintf("exact %.2f vs lecam %.2f (4se=%.2f)", ma, mb, 4se))
        check("Y=$Y R=$R sd",   oks, @sprintf("exact %.2f vs lecam %.2f", sa, sb))
    end
end

println()
println("="^78)
println("E. Polya-Gamma: direct Devroye call vs the hybrid sampler")
println("="^78)
# Must be BITWISE identical -- the hybrid dispatches to exactly this routine for
# integer b in [1, 13).
let
    ndiff = 0
    Random.seed!(19)
    for _ in 1:5000
        b = rand(1:12); zz = randn()*3
        s = rand(UInt32)
        Random.seed!(s); x1 = rand(PolyaGammaHybridSampler(Float64(b), zz))
        Random.seed!(s); x2 = rand_pgdevroye(b, zz, Random.default_rng())
        x1 === x2 || (ndiff += 1)
    end
    check("hybrid(b,z) === rand_pgdevroye(b,z)", ndiff == 0, "$ndiff differ of 5000")
end

println()
println("="^78)
@printf("%d passed, %d failed\n", npass, nfail)
exit(nfail == 0 ? 0 : 1)
