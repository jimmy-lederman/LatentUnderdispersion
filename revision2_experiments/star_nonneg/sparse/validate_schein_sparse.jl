# Schein joint-distribution validation of the STARMFNNsp Gibbs sampler at
# NON-conjugate hyperparameters, where the U and V updates are MH or slice
# rather than exact draws.
#
# Schein rather than Geweke deliberately: the successive-conditional Geweke
# chain is autocorrelated, which invalidates the i.i.d. z-test and produced
# false failures earlier in this project. scheinTest draws a fresh (data, state)
# from the joint for every replicate and applies nthin transitions with data
# fixed, so backward samples are i.i.d. under a valid kernel.
#
# This is the whole-model gate. validate_samplers.jl already checked the two 1-D
# conditionals against exact rejection draws; this checks that dropping them into
# the full sweep leaves the joint invariant.
#
# Usage: julia --project=../../.. validate_schein_sparse.jl [nsamples] [nthin]

include("../../../models/STARmodels/revision2/STARMFNNsparse.jl")
using Random, Statistics, Printf

nsamples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
nthin    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

function run_schein(a, c, sampler; N=6, M=5, K=2, nsamples=3000, nthin=10)
    model = STARMFNNsp(N, M, K, a, c, 0.01, 1.0, 1.0, identity, identity, sampler)
    Random.seed!(42)
    f, b = scheinTest(model, ["U_NK", "V_KM", "sigma2", "d"],
                      nsamples=nsamples, nthin=nthin)
    funcs = Dict(
        "U_NK" => [("var", x -> var(vec(x))), ("max", maximum), ("entry11", x -> x[1, 1])],
        "V_KM" => [("mean", x -> mean(vec(x))), ("var", x -> var(vec(x))), ("entry11", x -> x[1, 1])],
    )
    pvals = Float64[]; names = String[]
    for key in ["U_NK", "V_KM", "sigma2", "d"]
        if f[key][1] isa Number
            push!(pvals, gewekepvalue(Float64.(f[key]), Float64.(b[key]))); push!(names, key)
            push!(pvals, gewekepvalue(log.(Float64.(f[key])), log.(Float64.(b[key])))); push!(names, "log($key)")
        else
            for (fname, func) in funcs[key]
                push!(pvals, gewekepvalue([func(s) for s in f[key]], [func(s) for s in b[key]]))
                push!(names, "$fname($key)")
            end
        end
    end
    minp = minimum(pvals)
    bonf = min(1.0, minp * length(pvals))
    @printf("a=c=%.2f  %-6s  min p = %.4f  Bonferroni = %.4f  -> %s\n",
            a, sampler, minp, bonf, bonf > 0.05 ? "PASS" : "FAIL")
    bonf <= 0.05 && for (n, p) in zip(names, pvals)
        p < 0.05 && @printf("      offending: %-14s p = %.4f\n", n, p)
    end
    return bonf > 0.05
end

println("== whole-model Schein test, nsamples=$nsamples nthin=$nthin ==")
ok = Bool[]
push!(ok, run_schein(1.0, 1.0, :exact))          # control: the validated path
for ac in (0.5, 0.25, 0.1, 0.05, 0.01)
    push!(ok, run_schein(ac, ac, :mh))
    push!(ok, run_schein(ac, ac, :slice))
end
@printf("\n%d/%d configurations pass\n", sum(ok), length(ok))
