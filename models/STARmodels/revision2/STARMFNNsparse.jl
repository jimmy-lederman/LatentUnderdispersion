include("../../../helper/MatrixMF.jl")
using Distributions
using LinearAlgebra
using LogExpFunctions

# STAR-NN with GENERAL Dirichlet concentration a and gamma shape c.
#
# STARMFNN.jl is exact only at a = c = 1 and asserts it. That is not an
# implementation shortcut, it is where conjugacy stops:
#
#   U mass-shift on [0,s]:  p(t) prop. exp(-A t^2/2 + B t) * t^(a-1) (s-t)^(a-1)
#   V weight on (0,Inf):    p(t) prop. exp(-A t^2/2 + B t) * t^(c-1)
#
# with the gamma RATE already folded into B. At a = c = 1 both power factors
# vanish and the conditionals are truncated normals. Away from 1 they are not
# standard, and for a, c < 1 they diverge at the boundary -- exactly where a
# Gaussian proposal has no mass.
#
# This file is separate from STARMFNN.jl on purpose: that model is validated and
# used in the finished Section 6.3 sweep, and must stay bit-identical.
#
# Two samplers, selected by `sampler`:
#   :mh     independence Metropolis-Hastings proposing from the a = c = 1
#           truncated-normal conditional, accepting on the ratio of the power
#           factors. The natural implementation, and the one whose acceptance
#           rate degrades measurably as a, c fall.
#   :slice  slice sampling on the exact log conditional. Bounded shrinkage for
#           U; step-out on the right for V. Handles the boundary singularity,
#           at several density evaluations per update.
#   :exact  the a = c = 1 truncated-normal draw, for checking the samplers
#           reproduce it when the power factors are absent.
#
# Neither sampler is a strawman: :mh proposes from the best available closed
# form, and :slice needs no tuning. Report whichever does better.

struct STARMFNNsp{F1,F2} <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64        # Dirichlet concentration for columns of U_NK
    c::Float64        # gamma shape for V_KM
    d::Float64        # gamma rate for V_KM
    a0::Float64       # InverseGamma shape for sigma2
    b0::Float64       # InverseGamma scale for sigma2
    g::F1
    g_inv::F2
    sampler::Symbol   # :mh, :slice, or :exact
end

function STARlogpmf(model::STARMFNNsp, x, mu, sigma2)
    std = sqrt(sigma2)
    if x == 0
        z = (model.g(0) - mu) / std
        return logcdf(Normal(), z)
    end
    z1 = (model.g(x) - mu) / std
    z0 = (model.g(x - 1) - mu) / std
    hi = logcdf(Normal(), z1)
    lo = logcdf(Normal(), z0)
    return hi + log1mexp(min(lo - hi, -1e-12))
end

function evalulateLogLikelihood(model::STARMFNNsp, state, data, info, row, col)
    Y = data["Y_NM"][row, col]
    mu = dot(state["U_NK"][row, :], state["V_KM"][:, col])
    return STARlogpmf(model, Y, mu, state["sigma2"])
end

function sample_prior(model::STARMFNNsp, info=nothing)
    dhyper = sample_rate_prior()
    U_NK = rand(Dirichlet(fill(model.a, model.N)), model.K)
    V_KM = rand(Gamma(model.c, 1 / dhyper), model.K, model.M)
    sigma2 = rand(InverseGamma(model.a0, model.b0))
    return Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2, "d" => dhyper)
end

function forward_sample(model::STARMFNNsp; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    sigma2 = state["sigma2"]
    Z_NM = rand.(Normal.(Mu_NM, sqrt(sigma2)))
    Y_NM = zeros(model.N, model.M)
    for n in 1:model.N, m in 1:model.M
        Y_NM[n, m] = Z_NM[n, m] < 0 ? 0 : ceil(model.g_inv(Z_NM[n, m]))
    end
    state["Z_NM"] = Z_NM
    return Dict("Y_NM" => Y_NM), state
end

# ---------------------------------------------------------------- samplers

"Truncated draw from exp(-A t^2/2 + B t) on [lower, upper]; the a = c = 1 case."
function truncgauss(A, B, lower, upper)
    if A > 0 && isfinite(A) && isfinite(B)
        return rand(Truncated(Normal(B / A, sqrt(1 / A)), lower, upper))
    elseif B < 0
        return rand(Truncated(Exponential(-1 / B), lower, upper))
    elseif isfinite(upper)
        return lower + (upper - lower) * rand()
    else
        error("improper conditional in truncgauss: A=$A, B=$B on [$lower, Inf)")
    end
end

"""Slice sample on the bounded interval (lo, hi).

No stepping out is needed: the support is bounded, so (lo, hi) always brackets
the slice. For shape < 1 the density diverges at the endpoints, which shrinkage
handles -- the slice simply includes neighborhoods of both.
"""
function slice_bounded(f, t0, lo, hi; maxshrink=200)
    y = f(t0) + log(rand())
    isfinite(y) || return t0
    L, R = lo, hi
    for _ in 1:maxshrink
        t = L + (R - L) * rand()
        if f(t) > y
            return t
        elseif t < t0
            L = t
        else
            R = t
        end
    end
    return t0
end

"""Slice sample on (0, Inf).

The left end needs no stepping: for shape < 1 the density diverges at 0 so the
slice always reaches it, and for shape >= 1 zero is a valid bracket either way.
Only the right end is stepped out.
"""
function slice_positive(f, t0, w; maxstep=60, maxshrink=200)
    y = f(t0) + log(rand())
    isfinite(y) || return t0
    R = t0 + w
    for _ in 1:maxstep
        f(R) > y || break
        R += w
    end
    L = 0.0
    for _ in 1:maxshrink
        t = L + (R - L) * rand()
        if f(t) > y
            return t
        elseif t < t0
            L = t
        else
            R = t
        end
    end
    return t0
end

# ---------------------------------------------------------------- updates

function update_Z!(model::STARMFNNsp, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    sd = sqrt(sigma2)
    g0 = model.g(0)
    @views for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask) && mask[n, m] == 1
                z = rand(Normal(Mu_NM[n, m], sd))
                Y_NM[n, m] = z < 0 ? 0 : ceil(model.g_inv(z))
            end
            if Y_NM[n, m] == 0
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sd), -Inf, g0))
            else
                Z_NM[n, m] = rand(Truncated(Normal(Mu_NM[n, m], sd),
                                            model.g(Y_NM[n, m] - 1), model.g(Y_NM[n, m])))
            end
        end
    end
end

"U mass-shift update; returns (n_accepted, n_proposed) for the MH sampler."
function update_U!(model::STARMFNNsp, U_NK, V_KM, Z_NM, Mu_NM, sigma2)
    a = model.a
    nacc = 0; ntry = 0
    @views for k in 1:model.K
        for i1 in 1:(model.N - 1)
            i2 = model.N
            s = U_NK[i1, k] + U_NK[i2, k]
            s <= 0 && continue
            A = 0.0; B = 0.0
            for j in 1:model.M
                v = V_KM[k, j]
                e = Z_NM[i1, j] - (Mu_NM[i1, j] - U_NK[i1, k] * v)
                w = Z_NM[i2, j] - (Mu_NM[i2, j] - U_NK[i2, k] * v) - s * v
                A += 2 * v^2
                B += v * (e - w)
            end
            A /= sigma2
            B /= sigma2
            told1 = U_NK[i1, k]
            told2 = U_NK[i2, k]

            t = told1
            if model.sampler === :exact || a == 1.0
                t = truncgauss(A, B, 0.0, s)
            elseif model.sampler === :mh
                prop = truncgauss(A, B, 0.0, s)
                ntry += 1
                # proposal IS the a = 1 conditional, so it cancels exactly and
                # the ratio is only the power factors the proposal omits.
                #
                # The clamp is not cosmetic. truncgauss can return a value a
                # rounding error outside [0, s], and at a = 0.01 that made
                # log(s - prop) throw a DomainError on -1.8e-14. Below roughly
                # a = 0.05 the conditional's mass genuinely lies within machine
                # epsilon of the simplex boundary, so Float64 cannot represent it
                # faithfully however the update is written -- that is a property
                # of the target, not of this sampler.
                eps_s = 4 * eps(max(s, 1.0))
                if s <= 8 * eps_s
                    t = told1                       # interval too narrow to move in
                else
                    p1 = clamp(prop,  eps_s, s - eps_s)
                    t1 = clamp(told1, eps_s, s - eps_s)
                    lr = (a - 1) * (log(p1) + log(s - p1) - log(t1) - log(s - t1))
                    if isfinite(lr) && log(rand()) < lr
                        t = p1; nacc += 1
                    else
                        t = told1
                    end
                end
            else  # :slice
                f(x) = -A * x^2 / 2 + B * x + (a - 1) * (log(x) + log(s - x))
                t = slice_bounded(f, clamp(told1, 1e-12 * s, s - 1e-12 * s), 0.0, s)
            end

            U_NK[i1, k] = t
            U_NK[i2, k] = s - t
            for j in 1:model.M
                Mu_NM[i1, j] += (t - told1) * V_KM[k, j]
                Mu_NM[i2, j] += ((s - t) - told2) * V_KM[k, j]
            end
        end
    end
    return nacc, ntry
end

"V weight update; returns (n_accepted, n_proposed) for the MH sampler."
function update_V!(model::STARMFNNsp, U_NK, V_KM, Z_NM, Mu_NM, sigma2, drate=model.d)
    c = model.c
    nacc = 0; ntry = 0
    @views for k in 1:model.K
        for j in 1:model.M
            A = 0.0; B = 0.0
            for i in 1:model.N
                u = U_NK[i, k]
                e = Z_NM[i, j] - (Mu_NM[i, j] - u * V_KM[k, j])
                A += u^2
                B += u * e
            end
            A /= sigma2
            B = B / sigma2 - drate      # gamma RATE folds into the linear term
            vold = V_KM[k, j]

            vnew = vold
            if model.sampler === :exact || c == 1.0
                vnew = truncgauss(A, B, 0.0, Inf)
            elseif model.sampler === :mh
                prop = truncgauss(A, B, 0.0, Inf)
                ntry += 1
                # same guard as the U update: the support is (0, Inf), so only
                # the lower end can underflow
                p1 = max(prop, floatmin(Float64))
                v1 = max(vold, floatmin(Float64))
                lr = (c - 1) * (log(p1) - log(v1))
                if isfinite(lr) && log(rand()) < lr
                    vnew = p1; nacc += 1
                else
                    vnew = vold
                end
            else  # :slice
                f(x) = -A * x^2 / 2 + B * x + (c - 1) * log(x)
                w = A > 0 ? max(sqrt(1 / A), 1e-8) : max(vold, 1.0)
                vnew = slice_positive(f, max(vold, 1e-12), w)
            end

            V_KM[k, j] = vnew
            for i in 1:model.N
                Mu_NM[i, j] += U_NK[i, k] * (vnew - vold)
            end
        end
    end
    return nacc, ntry
end

function update_sigma2(model::STARMFNNsp, Z_NM, Mu_NM)
    resid = Z_NM .- Mu_NM
    return rand(InverseGamma(model.a0 + model.N * model.M / 2,
                             model.b0 + sum(resid .^ 2) / 2))
end

function backward_sample(model::STARMFNNsp, data, state, mask=nothing)
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    sigma2 = copy(state["sigma2"])
    Mu_NM = U_NK * V_KM
    Z_NM = zeros(model.N, model.M)

    dhyper = get(state, "d", model.d)
    update_Z!(model, Y_NM, Z_NM, Mu_NM, sigma2, mask)
    au, tu = update_U!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2)
    av, tv = update_V!(model, U_NK, V_KM, Z_NM, Mu_NM, sigma2, dhyper)
    sigma2 = update_sigma2(model, Z_NM, Mu_NM)
    # d | V stays conjugate for ANY shape c: Gamma(e0 + n*c, f0 + sum(V))
    dhyper = sample_rate_hyper(V_KM, model.c)

    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "sigma2" => sigma2,
                 "Z_NM" => Z_NM, "d" => dhyper,
                 "accU" => tu > 0 ? au / tu : NaN,
                 "accV" => tv > 0 ? av / tv : NaN)
    return data, state
end
