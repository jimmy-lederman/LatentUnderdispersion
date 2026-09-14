# The same Schein joint-distribution test as validate_joint.jl, but against the ORIGINAL
# committed sampler (git HEAD, with only the two changes it needs to run at all: the
# broken include paths and the undefined global `M`). This tells us whether a failing
# functional is pre-existing or was introduced by the rewrite.
#
#   julia -t 1 --project=. revision2_experiments/covid/validate_joint_orig.jl

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(@__DIR__, "reference/covid_orig.jl"))
using Random, Printf, Statistics, LinearAlgebra

N, M, K, Q, Dmax = 8, 6, 2, 3, 5
model = covid4orig(N, M, K, Q, Dmax,
                   1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                   0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
info = Dict("Y0_N" => reshape(collect(1:N), :, 1))

nsamples = 4000
nthin = 4
println("ORIGINAL sampler: N=$N M=$M K=$K Q=$Q Dmax=$Dmax, $nsamples replicates, $nthin sweeps")

Random.seed!(2024)
f_samples, b_samples = scheinTest(model,
    ["eps", "sigma2_tau", "sigma2_beta", "U_NK", "V_KM", "Tau_QM", "Beta_NQ", "D_NM"],
    nsamples=nsamples, nthin=nthin, info=info)

funcs = Dict(
    "eps"         => [("mean", x -> x),       ("sq",  x -> x^2)],
    "sigma2_tau"  => [("mean", x -> x),       ("log", x -> log(x))],
    "sigma2_beta" => [("mean", x -> x),       ("log", x -> log(x))],
    "U_NK"        => [("mean", x -> mean(x)), ("var", x -> var(vec(x)))],
    "V_KM"        => [("mean", x -> mean(x)), ("var", x -> var(vec(x)))],
    "Tau_QM"      => [("mean", x -> mean(x)), ("var", x -> var(vec(x))),
                      ("lag1", x -> mean(x[:,1:end-1] .* x[:,2:end]))],
    "Beta_NQ"     => [("mean", x -> mean(x)), ("var", x -> var(vec(x)))],
    "D_NM"        => [("mean", x -> mean(x)), ("var", x -> var(vec(x)))],
)

println()
@printf("%-14s %-6s %12s %12s %10s\n", "variable", "func", "forward", "backward", "p-value")
println("-"^58)
pvals = Float64[]; labels = String[]
for (key, fl) in sort(collect(funcs), by=first)
    for (fname, f) in fl
        fw = Float64[f(x) for x in f_samples[key]]
        bw = Float64[f(x) for x in b_samples[key]]
        p = gewekepvalue(fw, bw)
        push!(pvals, p); push!(labels, "$key.$fname")
        @printf("%-14s %-6s %12.5f %12.5f %10.4f%s\n",
                key, fname, mean(fw), mean(bw), p, p < 0.01 ? "  <-- LOW" : "")
    end
end
println()
println("low (<0.01): ", labels[pvals .< 0.01])
