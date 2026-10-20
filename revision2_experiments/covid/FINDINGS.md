# Section 7.2 (COVID-19) — sampler correctness, speed, and diagnostics

Date: 2026-08-09. Motivated by the same three referee threads that drove Sections 6.3 and
7.1: MCMC diagnostics (R1 2(d)), computing performance on real data (R1 2(b)), and the
"not enough iterations" concern. All timings measured on this machine (Apple M4 Pro,
Julia 1.11.2) with `revision2_experiments/covid/bench_before_after.jl`, which interleaves
old and new sweeps because absolute wall-clock drifts upward over a benchmarking session.

## TL;DR

1. **The committed §7.2 pipeline could not run at all.** Broken include paths in
   `models/covid/covid_final.jl` and all three ablation models; an undefined global `M`
   inside `backward_sample`; `runcovid_heldout.jl` type 4 including `ablation/covid3.jl`
   (which does not define `covid4`); `runcovid_full.jl` passing a `skipupdate=` kwarg
   that `fit` no longer accepts. All fixed.
2. **Two sampler bugs found, one pre-existing and one introduced-and-caught here.**
   - Pre-existing: the `tau_{q,t}` update ran under `@threads` even though its full
     conditional depends on both time neighbours — a data race, and a Jacobi rather than
     Gibbs sweep, so it did not leave the posterior invariant under threading. Now a
     sequential scan (see below: the expensive part is still parallel).
   - The `@threads` accumulators used `Threads.threadid()` under the default `:dynamic`
     schedule, which may migrate a task mid-iteration and corrupt the sufficient
     statistics. Now `:static` throughout.
3. **Speed: 5.15 s → 0.67 s per sweep, a 7.7x reduction in wall-clock**, and allocations
   3.09 GiB → 0.88 GiB per sweep. Split out: 1.49x from the serial work, the rest from
   threading that now actually scales (the old code's threading was both unsafe and
   bottlenecked on a serial block).

| | old | new | speedup |
|---|---|---|---|
| 1 thread | 5.15 s/sweep | 3.45 s/sweep | 1.49x |
| 8 threads | 1.57 s/sweep | 0.67 s/sweep | 2.34x |
| **old serial vs new 8-thread** | **5.15 s** | **0.67 s** | **7.7x** |
| allocations | 3.09 GiB/sweep | 0.88 GiB/sweep | 3.5x |

At 4 chains x 4000 sweeps that is 5.7 h/chain before and 0.74 h/chain after; a 10,000-sweep
chain is now under 2 h, which is what makes "run it longer" affordable.

## Where the time went, and what changed

Baseline decomposition (single thread, K=5, Q=25, Dmax=9, 918,784 cells):
Z-step 37%, D-update 35%, Polya-Gamma draws 19%, everything else 9%.

1. **D-update — the parent cdf was recomputed per candidate order.** Each cell called
   `logpmfOrderStatPoisson` five times, and each call re-evaluated `cdf(Poisson(mu), Y)`
   (an incomplete gamma) and re-acquired a **lock** on the shared coefficient-table cache
   — 4.6M lock acquisitions per sweep, which alone would have capped threading. Now
   `cdf`/`pdf`/`logpdf` are computed once per cell, the coefficient tables are built once
   per thread-chunk, and the lock-free `logpmf_orderstat_grid` is called directly. The
   log-probabilities are **bitwise identical** to the previous code (verified over 4048
   cases at each of Dmax = 5, 9, 15).
2. **Z-step thinning** — `rand(Multinomial(Z, P_K/sum(P_K)))` per cell allocated three
   objects; replaced by the shared `thin_multinomial!` into a per-thread buffer.
3. **Polya-Gamma** — `b = (Dmax-1)/2` is always a positive integer, so the hybrid sampler
   always dispatches to Devroye. Calling `rand_pgdevroye` directly hoists a struct
   construction and four string comparisons out of ~900k draws per sweep. Bitwise
   identical (verified over 5000 draws).
4. **`U_NK` update** — `sum(D_NM[n,:] .* V_KM[k,:])` per (n,k) is exactly
   `(D_NM * V_KM')[n,k]`; now one BLAS product instead of N*K length-M temporaries. A
   `Float64` copy of `D_NM` is made once per sweep so this and `U' * D_NM` both hit BLAS
   instead of the generic mixed Int/Float fallback.
5. **`tau` block** — the posterior precision `Beta' diag(w_m) Beta + lambda I` and the
   data part of the mean do not involve `Tau` at all, and they are essentially all of the
   arithmetic. They are now built AND Cholesky-factored in parallel over m; the
   sequential Gibbs scan that follows only adds the two neighbour terms and solves a
   25x25 triangular system per column. This is what unblocked threading — before it, the
   serial fraction capped the whole sweep at 2.3x on 8 threads.
6. **`tau`/`beta` draws** — `inv(A)` then `rand(MvNormal(mvec, V))` replaced by a draw
   through the Cholesky factor of the precision. Same distribution, no matrix inverse.
7. **Memory layout** — `W` and the Polya-Gamma pseudo-response are materialised in both
   orientations once per sweep, so the `tau` block (column walks) and the `beta` block
   (row walks) are both contiguous. The `beta` block's row walk of a 7 MB matrix, once
   per county, was a measurable share on its own.
8. **CRT** — `sampleCRT` is O(Y) with a Bernoulli per unit, applied to counts aggregated
   over all 3104 counties. Ported the le Cam tail approximation from the genes model.
   (Not hot at the prior draw, but it is unbounded in Y and cumulative counts are large.)

## Correctness evidence

- `validate_units.jl` — 32 checks. `update_D` log-probabilities bitwise identical to the
  previous implementation at Dmax = 5, 9, 15; identical D draws where the RNG stream also
  matches; `thin_multinomial!` against the exact multinomial expectation; the
  Cholesky-of-precision draw against `rand(MvNormal(inv(A)*rhs, inv(A)))` in mean and
  covariance; `sampleCRTlecam` against exact CRT in mean and sd; Polya-Gamma bitwise.
- `validate_blocks.jl` — the rewritten `tau_m` and `beta_n` draws against the original
  code's, for a fixed conditioning state, in mean and covariance.
- `validate_joint.jl` — Schein joint-distribution test. Drawing (theta, Y) from the prior
  and applying the kernel must leave theta's marginal unchanged; 17 functionals, no
  p-value below 0.01. **Run under `-t 1` and `-t 8`**: passing under threading is what
  certifies the `:static` schedule and the per-thread accumulators.
- `validate_joint_orig.jl` — the same test against the original sampler, so a failing
  functional can be attributed.

### The bug the joint test caught (worth reading before trusting a Gumbel-max anywhere)

The OddBinomial prior term was written `(d-1)*log(p)/2 + (Dmax-d)*log(1-p)/2`.
`p_{i,t} = logistic(beta_i' tau_t)` genuinely saturates to exactly 0.0 and 1.0 in Float64
for extreme cells, and there the zero exponent meets `log(0)`: `0 * (-Inf) = NaN`, not 0.
So the k=0 entry is NaN when p=0 and the k=half entry is NaN when p=1.

`argmax` propagates NaN and returns that index, which happens to be the correct candidate
in both limits — so the original code was accidentally right. Rewriting the Gumbel-max as
a scalar loop (to avoid allocating a 5-vector per cell) compares with `>`, which **skips**
NaN, so those cells silently fell through to D = 1. The effect was a ~2% downward bias in
E[D] — invisible in the unit tests, which never generated a saturated p, and caught only
by the joint-distribution test (p < 1e-4 on `D_NM.mean`, reproducible across seeds).

Fixed by guarding the zero exponents so the term is exact (`p^0 = 1`) and no NaN is
produced, which also makes the result independent of how the argmax is written. Saturated
p is now in the unit-test grid.

Note this also means an earlier "9.6x" speedup measurement in this session was inflated:
the bug was pushing cells to D = 1, and the sampler's cost is O(D). The 7.7x above is
measured against the corrected sampler.

## Diagnostics and reruns

- `analysis/covid/runcovid_heldout.jl` — rewritten to mirror `runflights_heldout.jl`:
  JIT warm-up before the timed fit, per-draw held-out log predictive density, per-cell
  posterior-mean density, mu on a fixed random subset of 200 cells, mean D per draw, and
  the environment (threads, Julia version, CPU, git commit, hostname) without which a
  timing column is not interpretable.
- `analysis/covid/summarize_covid.jl` — new. Information gain over the D = 1 baseline
  with SE over splits, plus rank-normalized split-Rhat and bulk/tail ESS
  (MCMCDiagnosticTools) on identifiable functionals only: the held-out log predictive
  density, mu on fixed cells, and mean D. Raw factor entries are excluded for the same
  reason as in Section 6.3 — theta and phi are identified only up to their scale
  trade-off.
- `MCMCDiagnosticTools` was **not in `Project.toml`**, so `summarize_flights.jl` could not
  run from a clean clone either. Added.

## Caveat on the timing numbers

The per-sweep figures above are measured two sweeps from the prior draw. The sampler's
cost is O(D) per cell, and D rises during burn-in, so a converged sweep is more expensive
than a warm-up sweep — the convergence run logs sweeps slowing from roughly 0.8 s to
1.3 s over the first 400. The old/new RATIO is measured on matched states and is sound,
but the absolute "hours per chain" figures should be re-measured from a converged state
before they go in the paper's timing column. Tracked as its own item.

## Decisions taken (author, 2026-08-10)

- **Dmax stays at 9.** The +0.061 nats that Dmax = 15 buys is not worth taking. The paper
  should state the choice and, ideally, that it was checked -- "9 was sufficient" is now an
  empirical claim rather than a default.
- **D is a computational device, not an estimand.** It buys a tighter conditional and better
  forecasts. Under that framing D does not need to mix and its ESS is not reported, which is
  what makes the binding-ceiling finding tolerable.
- **eps to be fixed at a constant** rather than inferred (best held-out density of the three
  arms tested, and it removes an unidentified parameter).
- **Reported diagnostics** are the held-out density and the factors, since the factors are
  interpreted in Figure 10.
- **start_V1 to be rescaled if it helps** -- under test.
- **Q may drop from 25 to 5 if performance holds** -- under test. It also directly addresses
  the over-parameterisation of the subset runs.

### What to report for the factors

The per-entry trajectories s_k*phi_{k,t} will not reach conventional Rhat. But Figure 10's
CLAIMS are much coarser than that object -- "factor 1 is the initial wave, factor 2 is flu
season" is a statement about WHEN a wave happened and HOW MUCH it accounts for, not about
phi at each of 295 days. Coarse functionals average over the noise:

| quantity | claim it supports | dimension |
|---|---|---|
| temporal centre of mass, sum t*phi_kt / sum phi_kt | when wave k happened | K |
| peak time, argmax_t phi_kt | same, ordinal and robust | K |
| duration (sd of the time distribution) | how long it lasted | K |
| activity share A_k / sum_j A_j, A_k = sum_t s_k phi_kt | how much of total growth wave k explains | K |
| state-level loadings sum_{i in state} theta_ik / s_k | which regions a wave hit | K x 50, not K x 3104 |

So the table becomes Rhat/ESS for the centre of mass, duration and activity share of each
factor, with the trajectories shown as figures carrying credible bands and an explicit note
that individual phi entries are not separately identified. These three are now computed by
`reported_diagnostics.jl`.

## Decisions taken (author, 2026-08-09)

1. **First column dropped.** Both run scripts now set `Y0_N = cumdf[:,1]` and
   `Y_NM = cumdf[:,2:end]`, giving N = 3104 counties and M = 295 modelled days, which is
   the T = 295 the paper states. Previously column 1 was both the conditioning value and
   the first modelled day, so t = 1 was conditioned on itself.
2. **`fit` gained an optional `skipupdate`,** forwarded to `backward_sample`. It defaults
   to `nothing`, in which case `backward_sample` is called exactly as before, so the
   Section 6.3 and 7.1 campaigns are untouched. Fixed-D runs of the covid model now pass
   `skipupdate = ["D_NM"]` alongside `constantinit`, which makes the D = 1 Poisson
   baseline **5.7x cheaper** — with D pinned to 1 the Z-step is trivial and the discarded
   D branch was 83% of the sweep, so the baseline was being charged roughly six times its
   real cost. Skipping is exact for everything reported: with D pinned, beta and tau have
   no data and nothing else in the model reads them.
3. **Hierarchical rate on theta implemented behind a flag** (`covid4(...; hier_b)`, with an
   18-argument constructor keeping the flat model as the default). Conjugate via
   `sample_rate_hyper`, the same Gamma(1,1) hyperprior as Section 6.3, and b is drawn from
   its own prior in `sample_prior` so chains start dispersed in it too. It passes the
   Schein joint-distribution test (19 functionals). Convergence was then tested on the
   real data — see the next section, which is the more important result.

## The mixing problem — this is the headline, not the hierarchical prior

`test_hier_convergence.jl`, 4 chains x 800 sweeps (600 burn-in, 200 kept), 300 counties
x 295 days, flat and hierarchical side by side from the same dispersed seeds:

| | flat Rhat | flat ESS | hier Rhat | hier ESS |
|---|---|---|---|---|
| eps | 3.81 | 4 | 3.30 | 5 |
| mean D | 4.28 | 4 | 4.34 | 4 |
| mu (200 cells, max) | 4.18 | 6 | 3.85 | 6 |
| total fitted mass | 3.41 | 5 | 2.95 | 5 |
| b | — (fixed) | — | 3.94 | 4 |
| mean theta | 3.81 | 4 | 4.17 | 4 |

**Neither model has converged**, on any functional, identified or not. Rhat is around 4
and bulk ESS is 4-6 out of 800 draws. Chain means differ by factors of 2-5 (eps spans
0.028 to 0.153 in the flat model; total fitted mass spans 4.8e5 to 9.8e5).

Two consequences:

1. **The hierarchical-prior question cannot be answered at this chain length.** The test
   was designed to show identified quantities converging while scale-only ones did not;
   instead nothing converges, so the ridge is neither confirmed nor ruled out. The one
   hint that runs against the ridge story is that the hierarchical model's total fitted
   mass is *more* consistent across chains than the flat model's (1.25x spread vs 2.0x),
   not less. Re-run the comparison once the chains are long enough to mean something.
2. **This is exactly referee R1 2(d), and it is worse than "we did not run enough
   iterations."** The paper's runs use 2000 burn-in. At 800 sweeps on a tenth of the
   counties the sampler is nowhere near stationarity, so simply scaling the sweep count
   may not be enough — the sampler may need a mixing change, not just more of it. The
   obvious candidate is the `tau` chain: it is currently a single-site scan over t, and a
   forward-filter backward-sample over the whole Gaussian AR(1) would update all T columns
   jointly. `phi` has the same structure. Neither is done here.

Note also that per-sweep cost on the FULL data grows several-fold during burn-in (roughly
0.8 s to over 4 s) as the inferred D climbs, and the subset run shows no such growth — so
the cost is concentrated in the large-count counties, where the latent-sum sampler draws
from Poissons with means in the hundreds of thousands. Any run-length plan has to budget
with the converged cost, not the warm-up cost.

### Which parameters mix, and why not (`diagnose_mixing.jl`)

4 chains x 4000 sweeps, last 2000 kept, 300 counties x 295 days, flat model, no scale move.
Three columns separate the three pathologies that all look like a large Rhat: single-chain
bulk ESS (low = slow mixing), drift = |mean(2nd half) - mean(1st half)|/sd within a chain
(large = not converged), and Rhat (large with good ESS and low drift = non-identified).

| group | quantity | ESS/chain | drift | Rhat |
|---|---|---|---|---|
| scalar | eps | 5.0 | 1.72 | 2.92 |
| scalar | sigma2_tau | 8.4 | 1.62 | 3.07 |
| scalar | sigma2_beta | 10.2 | 1.35 | 2.64 |
| scalar | mean D | 4.5 | 1.74 | 3.86 |
| identified | mu (150 cells) | 16.0 | 1.78 | 2.87 |
| identified | rho (150 cells) | 29.2 | 1.97 | 6.26 |
| identified | D (150 cells) | 314.1 | 1.97 | (artifact) |
| factor raw | theta | 12.5 | 1.86 | 3.82 |
| factor raw | phi | 17.6 | 1.90 | 4.19 |
| factor raw | s_k | 5.9 | 1.76 | 3.14 |
| factor scaled | theta/s_k | 13.6 | 1.85 | 3.35 |
| factor scaled | s_k*phi | 19.5 | 1.88 | 3.87 |
| factor aligned | theta/s_k | 13.6 | 1.85 | 3.76 |
| factor aligned | s_k*phi | 19.5 | 1.88 | 3.89 |

**Identifiability is not the binding constraint.** Normalising by the per-factor scale
barely moves Rhat (3.82 -> 3.35 on theta) and permutation alignment moves it back up. If
the scale ridge and label switching were driving this, those rows would drop sharply.

**Nothing has converged.** Drift is 1.35-1.97 on every quantity: each chain's own first
and second halves differ by ~1.8 sd after 2000 burn-in sweeps. That is not chains
disagreeing with each other, it is each chain still systematically moving. Per-chain ESS
of 5-30 out of 2000 implies an autocorrelation time of roughly 65-400 sweeps.

**Consequence for the reporting plan.** Reporting ESS on identified functionals does not
rescue this: mu, the quantity the analysis leads with, has per-chain ESS 16 and drift 1.78.
This is a sampler problem, not a presentational one.

Caveats: the D Rhat is an artifact (D is constant within a chain at some sampled cells, so
the between/within ratio blows up) — the per-chain ESS of 314 is the real number there and
says D flips freely within a chain while chains disagree on its level. All of this is a
300-county subset; full-data mixing may differ.

### The cause: initialisation, not the kernel (`synth_ladder.jl`, `synth_truthinit.jl`)

Synthetic data generated from the model itself, truth known, N=100 x M=60, K=3.

`synth_ladder.jl` varied the offset-to-growth ratio, the wave overlap, and K. Recovery is
fine when the growth term is a few percent of the level (relative error 0.13-0.29, 95%
coverage 87-89%, factor cosine 0.77-1.00 — **overlapping waves recover fine**, so
continuous factor trading is NOT the problem) and fails completely when the level dominates
(relative error 0.99, coverage 0%). Over-specifying K changed nothing.

Two things then ruled out the alarming explanations:
- The joint-distribution test **passes at Y0 = 20000**, so the kernel is correct at
  realistic count magnitudes. This is not a numerical bug in the deep tails.
- Started AT the truth, chains stay there. Started from the prior, they never leave zero.

So it is burn-in, and the mechanism is specific: the augmentation allocates latent counts to
each component in proportion to its CURRENT share of the rate. With phi drawn from
Gamma(start_V1 = 0.01, 1) the growth term starts near 0.01 while the rate is ~2e4, so the
factors receive essentially no counts and cannot grow. Self-reinforcing, and it only exists
when there is a large offset in the rate — which is why Sections 6.3 and 7.1 are unaffected
and why the "initialise everything from the prior" convention is wrong for this model.

Four initialisations, 4 chains x 3000 sweeps, truth mean growth 32.0:

| start | growth relerr | cov95 | eps (true 2.0) | per-chain mean growth, first kept -> last |
|---|---|---|---|---|
| prior draw | 0.99 | 0% | 2.60 | 0.4->0.6  0.1->0.1  0.0->0.1  0.0->0.0 |
| phi seeded at 1 | 0.57 | 49% | 2.91 | 8.3->16.2  5.5->11.7  7.2->16.1  7.6->16.4 |
| **scale-corrected** | 0.64 | **70%** | 0.85 | 31.0->31.9  25.3->30.6  28.6->31.3  29.3->30.8 |
| NMF per chain | 0.86 | 59% | 0.33 | 37.3->34.8  36.5->33.5  36.3->35.0  38.5->38.5 |
| at the truth (oracle) | 0.73 | 61% | 2.40 | 32.9->33.8  35.2->33.1  34.5->34.7  33.1->31.7 |

ADOPTED: **scale correction**. The prior draw's structure is kept and only its overall
magnitude is corrected, so chains start dispersed (25-31) and converge together (31-32) —
which is what makes Rhat meaningful. It reaches the truth more closely than the oracle start
and has the best coverage. phi = 1 escapes the trap but is still climbing at 3000 sweeps.
NMF is worse: it fits the whole increment, so the factors absorb eps and eps collapses to
0.33. `fit` now accepts a Function as an `init` value so the seed can depend on the draw.

Ignore the relerr column when comparing the last three rows: it is a per-cell relative error
and the truth contains near-zero cells, so it penalises an unbiased noisy estimate more than
a uniformly low one. The trajectories are the honest signal.

Two things this leaves open:
- **Coverage tops out at 70%, not 95%.** RESOLVED — see below. It is residual
  non-convergence, and it responds to run length.
- **eps is determined by its initialisation** (2.90 / 0.85 / 0.33 / 2.40 across arms against
  a truth of 2.0). It mixes too slowly to move from wherever it starts. Direct evidence for
  not interpreting eps.

**Unresolved and important for the paper:** the previous covid runs seeded phi at 1 (via
`constantinit`, which seeded rather than pinned before this revision) with ~5000 sweeps. In
this synthetic that regime is still climbing at 3000. Whether the published Section 7.2
results are affected has NOT been checked — the real data is 3104 x 295, ~50x more cells
informing the same factors, so it may climb much faster. One long chain on the real data,
watching whether the growth term plateaus, would settle it.

### The 70% coverage: residual non-convergence (`synth_coverage.jl`)

First hypothesis, WRONG: that it was prior-truth mismatch. A credible interval only has
nominal coverage averaged over truths drawn from the prior, and the hand-set truth (phi in
the tens) is unreachable under phi_1 ~ Gamma(0.01, 1). But drawing the truth from the very
prior being fitted still gives 68%, and the deliberately MISMATCHED arm gives 81% — higher.
So the prior is not the explanation for this symptom, and `start_V1` is not the culprit here
(it remains the cause of the initialisation trap above, which is a separate issue).

Bias is 10-20% of the interval width in every arm, so the intervals are too NARROW rather
than misplaced. With a kernel that passes the joint-distribution test, that points at
unexplored posterior. Confirmed by varying run length on a prior-typical truth, where a
correct converged sampler must reach 95%:

| | bias | median CI width | cov95 |
|---|---|---|---|
| 3000 sweeps | +5.56 | 31.4 | 68% |
| 3000 sweeps + scale move | +5.17 | 28.3 | 74% |
| 9000 sweeps + scale move | +2.99 | 29.6 | 81% |
| 27000 sweeps + scale move | +2.68 | 32.0 | 86% |

Monotone, but with sharply diminishing returns and a bias that flattens near +2.7 rather
than heading to zero — so run length alone is NOT the whole story, and an earlier claim in
this document that the coverage gap was "resolved" as pure non-convergence was premature.

The other half is eps. eps cannot move from wherever it is initialised, and the
scale-corrected init sets the growth target to mean(increment) - eps, so an eps that lands
low forces the growth term high by the deficit. Initialising eps at its generating value
(an oracle DIAGNOSTIC, not a fix):

| | bias | cov95 |
|---|---|---|
| eps from prior, 3000 | +5.17 | 74% |
| eps at truth,   3000 | +3.76 | 75% |
| eps from prior, 9000 | +2.99 | 81% |
| eps at truth,   9000 | **+1.87** | **87%** |

At matched length, fixing eps cuts the bias 37% and adds 6 points of coverage, and it
improves faster with length as well. **eps at truth with 9000 sweeps (87%) matches eps from
prior with 27000 (86%)** -- so fixing eps is worth roughly 3x in run length.

Two hypotheses for this symptom were tested and FAILED, which is worth recording so they are
not retried: prior-truth mismatch (the deliberately mismatched arm scored HIGHER, 81% vs
68%), and eps alone (at 3000 sweeps eps-at-truth moved coverage by one point).

The scale move helps a little and does so by narrowing the interval while raising coverage,
i.e. by correcting location rather than adding spread; it is free, so it is left on.

**Consequence: calibrate the run length by simulation-based coverage, not by Rhat or ESS.**
The ladder above showed Rhat/ESS are actively misleading on this model — the arms that
looked worst by Rhat gave the best inference, and the arms that looked converged were the
ones with 0% coverage. Coverage on a prior-typical synthetic truth says what it means:
generate from the model, fit, check that 95% intervals cover 95%. That is a defensible
answer to "how many iterations is enough" in a way "we ran until Rhat < 1.01" would not be
here, and it directly addresses R1 2(d).

### Dmax IS BINDING — the reported order reflects the bound, not the data (`dmax_ladder.jl`)

Real 200-county subset, 2 chains x 15000 sweeps, scale-corrected init + scale move:

| | mean D at 15000 | frac of cells AT Dmax | growth | max Rhat(mu) |
|---|---|---|---|---|
| Dmax = 9 | 8.143 | **88.0%** | 12.79 | 1.95 |
| Dmax = 15 | 12.854 | **82.1%** | 12.49 | 2.27 |

Raise the ceiling and the inferred order follows it. The growth term is unchanged either
way (12.5 vs 12.8), so the fitted RATE is robust — it is D specifically that is censored.
mean D was still creeping up at the final checkpoint in both (+0.4 to +1.8%).

This is the same shape as the MedNB dispersion-floor issue recorded for Section 6.3, and
more consequential, because D is what Section 7.2 interprets.

**Mechanism, measured** (`d_by_increment.jl`, 150 counties, 3000 sweeps, Dmax = 9): the
posterior mean of D falls monotonically with the size of the observed increment.

| bucket | cells | share | mean D | frac near Dmax |
|---|---|---|---|---|
| inc < 0 (corrections) | 662 | 1.5% | 7.683 | 73.9% |
| inc == 0 | 16609 | 37.5% | 7.606 | 67.8% |
| inc == 1 | 4708 | 10.6% | 8.017 | 78.1% |
| inc 2-5 | 7836 | 17.7% | 7.833 | 75.8% |
| inc 6-20 | 7827 | 17.7% | 7.311 | 69.0% |
| inc 21-100 | 5252 | 11.9% | 6.058 | 51.8% |
| inc > 100 | 1356 | 3.1% | **3.049** | 17.8% |

Cells with increments <= 5 -- 66% of the data -- sit near the ceiling; cells with large
increments settle at D ~ 3, comfortably inside it. The reason is the paper's own thesis: the
rate is mu ~ Y_{i,t-1}, so the Poisson noise is sqrt(mu) -- +/-100 for a county at 10,000
cases -- while the actual day-to-day increment is often 0 or 1. Placing mass on an
observation that close to the rate needs concentration of order mu/inc^2, so small
increments demand a large order. **D is quantifying how far below Poisson the real
dispersion is, and Dmax caps how much underdispersion the model can express.**

So the reported D is a FLOOR on the effect rather than an estimate of it. That is a
supportive finding for Section 7.2's argument, not a damaging one -- but it does mean the
inferred order cannot be read as "the data's order", and the honest framing is D as a
computational device bought for predictive gain. Under that framing D need not mix and its
ESS is not a reported quantity; what must hold is that the PREDICTIVE claim is stable and
that the ceiling does not cost forecast accuracy (`d_predictive.jl`).

It also explains eps collapsing to 0.004 on real data: pushing eps down keeps mu just above
Y_{i,t-1}, so the median of the rate lands on the observation.

(The `frac at D=1` column of that script's output is wrong — it prints impossible values.
The conclusion does not rest on it; `mean D` and `frac at Dmax` are mutually consistent.)

### Predictive consequences of the ceiling (`d_predictive.jl`)

200 counties, one forecasting mask, 2 chains x 4000 sweeps. Held-out information rate:

| arm | inforate | gain over Poisson | fit time |
|---|---|---|---|
| D = 1 fixed (Poisson) | -4.7649 | — | 0.7 min |
| D = 9 fixed | -4.4744 | +0.2905 | 8.6 min |
| D inferred, Dmax = 9 | -4.3925 | +0.3724 | 12.0 min |
| D inferred, Dmax = 15 | -4.3310 | +0.4338 | 17.1 min |

- **inferred(9) - fixed(9) = +0.0819.** Heterogeneity earns its keep despite 88% of cells
  being pinned at the ceiling: the value is in the MINORITY of large-increment cells, which
  need a LOW order (D ~ 3) that a fixed-D model cannot give them. That answers the obvious
  objection to Eq. (13).
- **inferred(15) - inferred(9) = +0.0614.** The ceiling costs prediction — comparable to
  what the heterogeneity itself buys, and ~16% of the total advantage over Poisson. Mean D
  reaches 9.288 at Dmax = 15, above the old ceiling. So "why Dmax = 9" needs either a raise
  or an explicit statement of the cost. Gains are diminishing (1->9 buys 0.29, 9->15 buys
  0.06), so a saturation argument is available for whatever value is chosen.

Also note the fit-time column: cost grows roughly linearly in D, because the latent-sum
sampler takes D sequential steps per cell. Useful context for the R1 2(b) compute table.

CAVEAT: one mask, 200 counties, 2 chains, 4000 sweeps. Indicative only; a number for the
paper wants replication across the five mask seeds at full size.

### The eps Metropolis step: implemented, validated, DID NOT DELIVER

| | bias | cov95 |
|---|---|---|
| eps Gibbs only, 3000 | +4.92 | 76% |
| eps + MH step, 3000 | +5.97 | 65% |
| eps at truth, 3000 (oracle) | +3.31 | 76% |
| eps + MH step, 9000 | +2.69 | 83% |
| eps at truth, 9000 (oracle) | +2.43 | 81% |

At 9000 it matches the oracle — but so does plain Gibbs (81% in the run-length ladder). At
3000 it is worse than doing nothing. The oracle experiment predicted 3x in run length; the
actual move does not realise it. It is implemented behind `covid4`'s `eps_mh` flag, passes
the Schein joint test (so it does not perturb the target), and is OFF by default.

### Caveat that qualifies several numbers above

The same configuration (eps at truth, 9000 sweeps) scored 87% in one run and 81% in
another. The only difference was the thread count (6 vs 4), which changes the per-task RNG
streams. **These coverage estimates carry roughly +/- 5-6 points of Monte Carlo noise.** The
run-length trend (74 -> 81 -> 86) spans enough to survive that; finer distinctions —
the scale move's 68 -> 74, eps's 81 -> 87 — do not, and should not be relied on.

### Diagnostics on the REPORTED quantities, and five failed interventions

Report Rhat as median/p90/max over parameters, never the max alone: quoting the max over
150 cells reads as "nothing converged" when the median parameter is far better. An earlier
version of this document made exactly that error.

100 counties, 4 chains, 6000 sweeps, scale-corrected init + scale move:

| window | quantity | Rhat med/p90/max | ESS/chain |
|---|---|---|---|
| 1000-2000 | held-out log density | 1.98 | 4.7 |
| | mu (150 cells) | 1.62/2.33/3.66 | 11.0 |
| | theta/s_k aligned | 1.75/2.60/3.79 | 11.5 |
| | s_k*phi aligned | 2.49/3.24/3.88 | 14.9 |
| 4000-6000 | held-out log density | 2.12 | 5.9 |
| | mu (150 cells) | **1.41**/2.03/2.87 | 17.6 |
| | theta/s_k aligned | 1.68/2.54/3.30 | 17.2 |
| | s_k*phi aligned | **2.52**/3.26/4.02 | 23.4 |

**mu converges slowly** (median 1.62 -> 1.45 -> 1.41). **The decomposition does not**:
theta/s_k is flat at ~1.68 and s_k*phi -- exactly what Figure 10 plots -- is flat at ~2.5
and is the worst-behaved quantity in the model. A well-determined mu admits many equally
good factorisations, so this is factor identifiability, not sampler mixing on mu.

**A measurement error found by the author's question, and fixed.** The relabelling was
per-CHAIN: match each chain's MEAN factor profile to chain 1's mean, apply that one
permutation to every draw. That is only valid if labels are stable within a chain, and they
are not -- label switching across draws inside a chain is standard, which smears the chain
mean and makes the match meaningless. `align_draws` now relabels EVERY DRAW against an
iterated consensus reference (Stephens-style). Effect at 4000-6000: s_k*phi median
2.52 -> 2.27, and 2.49 -> 2.04 in the early window where switching is commoner. Real, but
does not rescue the factors.

NOTE: the K=3 comparison below was measured with the OLD per-chain relabelling, so its
verdict rests on a biased metric. The bias is ~10% and the K=3 gaps were larger, so the
conclusion probably survives -- but it is not settled.

Six interventions tested; none moved the factor diagnostics:

| intervention | result |
|---|---|
| eps marginal Metropolis step | no gain; worse at short runs |
| removing eps entirely | no gain (and eps was never doing its "anchoring" job: min phi ~1e-281 in every arm) |
| longer runs | s_k*phi flat at ~2.5 across 6000 sweeps |
| reducing K to 3 | slightly WORSE on every quantity (mu 1.49, theta 1.90, s_k*phi 2.60, held-out 3.42) |
| scale move (PX-DA) | small gain only |
| per-draw relabelling (a measurement fix, not a model change) | s_k*phi 2.52 -> 2.27; real but not a rescue |
| sparsity-inducing theta shape a = 0.2 (was 1.0) | wash: s_k*phi 2.27 -> 2.17 but mu 1.41 -> 1.67 |

The `a < 1` test was aimed at SEPARABILITY, which is what makes a non-negative
factorisation identifiable: if every county loads on all K waves rather than a few, there
is a cone of equally good solutions and no amount of sampling picks one out. Pushing prior
mass toward zero did sharpen phi slightly and blunt mu slightly. Not a win.

HMC is ruled out by the author on principle: the paper's contribution is that the
order-statistic augmentation yields closed-form Gibbs, so reaching for gradient-based
sampling in the flagship case study would undercut the thesis.

**The one untested variable is N, and the subsets are not merely a scaled-down version of
the real problem.** Every factor diagnostic above is on 100-200 counties. Two reasons that
matters: phi_{k,t} is informed by ALL counties simultaneously, so the full 3104 gives
15-30x more information about exactly the quantity that will not converge; and these runs
keep Q = 25, so the Polya-gamma layer alone is beta at 100 x 25 = 2500 parameters for
29,500 cells -- badly over-parameterised at subset size, proportionate at full size. Six
interventions were tuned against a differently-conditioned problem. Check N before
concluding anything about the factors.

Fallback position if it is not an artifact: report held-out density and mu with their
diagnostics and an explicit statement that Rhat has not reached conventional thresholds;
present the factors with cross-chain agreement shown rather than as point estimates; and
note that the predictive claim rests on the information rate, which was stable and
correctly ordered across every arm run here.

### Two separate problems, finally separated (N = 300, 2000-4000 window)

| quantity | with offset | no offset (increments) |
|---|---|---|
| mu | Rhat 1.54, ESS 15.9 | Rhat 1.88, ESS **48.5** |
| theta/s_k | 1.43, ESS 19.4 | 1.75, ESS **58.8** |
| s_k*phi | 2.17, ESS 17.3 | 2.12, ESS **41.5** |
| FACTOR centre of mass | 3.09, ESS 3.9 | 2.94, ESS 2.8 |
| FACTOR duration | 3.39, ESS 2.3 | 2.51, ESS 7.8 |
| FACTOR activity share | 3.51, ESS 2.7 | 2.90, ESS 3.4 |

1. **Mixing efficiency IS caused by the offset.** ESS roughly triples when the model is
   refitted to increments. The mechanism is information, not augmentation noise: the
   observation is a median-of-D Poisson with rate mu, so its noise scales with sqrt(mu),
   and mu is dominated by Y_{i,t-1}. With the offset, 0.47*sqrt(1200) = 16.3 of noise
   against a growth signal of ~12.8 gives per-cell SNR ~0.8; without it, 0.47*sqrt(12.8)
   = 1.7 against the same signal gives ~7.6. The cumulative formulation buys robustness to
   reporting corrections and pays about 3x in sampling efficiency.
2. **Between-chain disagreement is NOT caused by the offset.** Rhat barely moves, and mu
   and theta actually get slightly worse. It persists in the plain increments model, which
   is the ordinary Poisson-factorisation regime. This is intrinsic factor non-identifiability
   for this data at K = 5, and it would be present in any factor model of these counts.

This is the better outcome for the paper: the modelling choice is responsible for an
efficiency cost, which sweeps can buy back, and NOT for the non-identifiability.

**An earlier version of this section gave a wrong mechanism** -- it compared the factor
share of the latent total Z (~90) against the sampling noise in Z (~92). That is the wrong
comparison: the count allocated to factor k is Binomial(Z, p_k) with p_k proportional to
theta*phi/mu, so its variance is ~D*theta*phi, the plain Poisson noise for that component,
and the offset does not inflate it. The correct channel is the likelihood's information, as
above.

### Also settled

- **Q = 5 buys nothing.** Neither convergence (slightly worse on every quantity at N=300)
  nor compute (4.6-5.3 min/chain vs 4.5-5.1 for Q=25). Expected in hindsight: the tau and
  beta blocks were ~3% of a sweep, and the Z-step and D-update do not depend on Q. Its
  predictive cost was not measured. Keep Q = 25 unless there is a parsimony reason.
- **Coarse factor summaries are WORSE, not better.** Centre of mass / duration / activity
  share sit at Rhat 3-3.5 with ESS 2-4, against ~2.2 for the per-entry trajectories. The
  per-entry number is a MEDIAN over 1475 entries, most of them near-zero phi values that
  trivially agree, which dilutes; the coarse summaries are 5 global functionals with no
  dilution. They are the stricter and more honest measure, and the proposal to report them
  as a rescue does not work.
- **N helps a little.** N=100 -> N=300 at the same window: theta 1.66 -> 1.43, phi
  2.41 -> 2.17, mu 1.45 -> 1.54. Real but modest; the slope does not reach 1.01.

### Candidate fixes, and their status

- **Scale move on (theta_{:,k}, phi_{k,:})** — IMPLEMENTED and validated (`scale_move!`,
  enabled by covid4's `scale_move` field). Exact MH on the same target: the complete-data
  likelihood is invariant under theta -> kappa*theta, phi -> phi/kappa because
  sum_i Y_NK[i,k] = sum_t Y_MK[t,k], so the ratio is prior x Jacobian only. Passes the
  Schein joint test. A probe on real data collapsed a 3.5x between-chain disagreement in
  sum_i theta_{i,1} to 9%, acceptance 0.63. NOT yet tested against the table above.
- **Data-informed initialisation** — NOT tried, and probably the largest single win.
  phi starts from Gamma(0.01, 1) (mean 0.01) while daily growth is in the hundreds, so the
  factors must climb orders of magnitude through a gamma-Markov chain that takes small
  steps. Initialisation is not part of the target, so warm-starting theta/phi from an NMF
  of the increments is free of inferential cost.
- **Marginal MH step on eps** — NOT implemented, and now the highest-value remaining item.
  eps is slow because the augmented-data conditional Gamma(g + sum_noise, h + sum D) is
  ~0.15% wide while the marginal is much wider. An MH step using the actual likelihood with
  the latent counts integrated out would let it move by its marginal width; cost is one
  full-data likelihood evaluation per proposal (~0.03 s threaded, so 3-5 proposals adds
  15-25% to a sweep). The oracle experiment above measures the payoff: **worth roughly 3x in
  run length**, which dwarfs the 15-25% overhead. Pure sampler change, no model or text
  implications.
- **Reduce K, chosen by held-out density** — NOT tried. If K = 5 is more factors than the
  data distinguishes, the trading and the slow mode are both symptoms. Needs only a grid
  run with the existing harness.
- **FFBS for tau** — DROPPED. tau and beta are not interpreted, so their mixing does not
  matter.
- **Dirichlet prior on theta columns** — RULED OUT by the author: Y_{i,t-1} + eps sits in
  the rate alongside theta'phi, so the Poisson-multinomial argument that would give
  Dirichlet conjugacy does not go through.

## Open decisions (not made here)

1. **Whether to adopt the hierarchical theta rate**, pending the convergence run above.
   If b and the raw theta scale fail to converge while mu does, the honest conclusion is
   that the pooling is not identified here and the flat prior stays — which is still a
   reportable answer to the referee, and a sharper one than not having tried.
2. **The county count.** The paper says C = 3,004; the shipped data has 3,104 rows. Only
   one of the two is right and it is not a code question.
3. **Run length and the campaign grid.** Target ESS should be set the way Section 6.3 set
   it (>= 1000 on the reported functional) rather than by a fixed sweep count, and the
   grid of (mask, chain, D) needs deciding before anything is submitted. `configs.sh` /
   `submit_all.sbatch` equivalents for covid do not exist yet.
4. **Threads on the cluster.** Section 7.1 deliberately ran single-threaded so that
   flights and flights_STAR were comparable. Section 7.2 has no cross-model threading
   asymmetry — every row is the same code — so threading is fair here, but the thread
   count has to be fixed across the whole campaign and recorded (the run script already
   saves it).

## THE FULL-MATRIX ANSWER (cluster job 1358460, run 2026-08-11, analysed 2026-09-11)

4 chains x 15000 sweeps on the FULL matrix (N = 3104, M = 295, K = 5, Q = 25, Dmax = 9),
eps fixed at 0.5, scale-corrected init, scale move on. AMD EPYC, 16 threads,
OPENBLAS_NUM_THREADS=1, 1.08-1.39 s/sweep, 4.5-5.8 h per chain.

| window | quantity | Rhat med/max | ESS med/min/pooled |
|---|---|---|---|
| 7500-15000 | held-out log density | 2.72/2.72 | 12/5/17 |
| | mean D | 3.09/3.09 | 4/4/16 |
| | mu (cells) | **1.22**/3.63 | 25/5/38 |
| | theta/s_k aligned | 1.51/3.16 | 20/4/25 |
| | s_k*phi aligned | 2.34/4.24 | 10/4/18 |
| | FACTOR centre of mass | 2.82/3.51 | 4/4/16 |
| | FACTOR activity share | 2.96/4.45 | 4/4/16 |

**The N hypothesis was half right.** mu's median Rhat improved from 1.54 on the 300-county
subset to 1.22 on the full matrix -- the best figure obtained anywhere -- consistent with
more counties pinning the fitted surface. theta (1.43 -> 1.51) and phi (2.17 -> 2.34) did
NOT improve. More data sharpens the surface, not its decomposition.

**Nothing reaches a reportable threshold, and it is not trending there.** Pooled ESS is
16-38 out of ~1875 retained draws per window. Across the three windows mu goes
1.29 -> 1.31 -> 1.22, phi 1.71 -> 2.52 -> 2.34, held-out density 2.04 -> 3.66 -> 2.72 --
noise around a plateau, not convergence in progress. Doubling the sweeps again would not
change the answer.

CONCLUSION: the Section 7.2 sampler does not produce conventionally converged chains on the
full data at any run length worth buying. This is measured on the real problem, not inferred
from subsets. The honest reporting position stands: report held-out density and mu with
their diagnostics and an explicit statement that Rhat has not reached conventional
thresholds; present the factors as exploratory with cross-chain overlays rather than point
estimates; rest the predictive claim on the information rate, which was stable and correctly
ordered across every arm tested.

Raw functionals: output/covid/diag/diag_chain{1..4}.jld (45 MB each), also on the cluster at
/net/projects/schein-lab/jimmy/LatentUnderdispersion/output/covid/diag/.
Regenerate the table with: julia --project=. revision2_experiments/covid/cluster/analyze_diag.jl

## The pairwise shear move, and what its failure revealed (2026-09-11)

Built `shear_move!` -- the off-diagonal partner of the scale move. For factors (j, k) and
scalar a, theta_{:,k} <- theta_{:,k} + a*theta_{:,j} and phi_{j,:} <- phi_{j,:} - a*phi_{k,:}
is [theta_j theta_k]*A, A^{-1}*[phi_j;phi_k] with A = [1 a; 0 1]: every mu_{i,t} is EXACTLY
unchanged, det A = 1, so the Metropolis ratio is the prior ratio alone. Together with the
scale move this spans the 2x2 mixing group under which a non-negative factorisation is not
identified. Passes the Schein joint test. Proposal is uniform on the feasible (non-negativity)
interval, which is symmetric because a shear is a translation along that line.

**It never accepts, and the reason is the finding.** Decomposed at a fitted state on the
100-county subset, 40 proposals per ordered pair:

| | median | p10 | p90 |
|---|---|---|---|
| feasible width of a | 0.0066 | | |
| theta prior log-ratio | +0.37 | +0.01 | +1.93 |
| **phi prior log-ratio** | **-211** | **-411** | **-52** |
| fraction with total ratio > log(0.5) | 0.000 | | |

The theta prior slightly FAVOURS shears. The c = 100 gamma-Markov prior on phi rejects every
one by 50-400 nats -- and not because the shear is large (the cone only allows 0.7%): the
feasible boundary sits exactly where phi_j is smallest relative to phi_k, so the shear pushes
phi_j toward zero there and the prior's (alpha_t - 1)*log(phi_t) term with alpha_t ~ 100+
punishes it by hundreds of nats.

**Consequence: the geometry was misdiagnosed.** The likelihood is flat along shears, but the
prior curves it SHARPLY. The factorisation is identified BY THE SMOOTHNESS PRIOR, and the
landscape over factorisations is RUGGED -- several sharp local modes (different smooth 5-row
splits of the same growth surface) separated by valleys where any intermediate split is
rough. Chains fall into different modes and cannot cross. This fits everything: Rhat stuck,
ESS tiny, no mu-preserving move can help, and the earlier "not multimodal" verdict -- which
was measured on mu, not on the factorisation.

It also explains why the eight interventions failed: none of them changed the ruggedness.

**The direct test is c.** At c = 100 the prior expects phi to move ~6% per day; at c = 10,
~20%. If the modes merge and factor Rhat collapses at c = 10, the problem is prior strength,
and the question becomes whether c = 100 is essential (it was likely chosen for visually
smooth waves). `reported_diagnostics.jl` takes c as ARGS[10]. Running c = 100 vs c = 10 on
the 100-county subset, otherwise identical.

If weakening c is unacceptable, the principled sampler answer for a rugged landscape is
parallel tempering on c -- chains at c = 100, 30, 10, 3 with swaps, so the loose chain crosses
modes and swaps carry that to the tight one. Still data augmentation throughout.

Side note: two `reported_diagnostics.jl` jobs launched in parallel at -t 4 each ran at
~2 s/sweep, 30x slower than the same code measures alone (0.068 s/sweep). Cause not found;
killed. Run these one at a time at -t 8.

### Julia parallel-GC segfault (operational, 2026-09-11)

A `reported_diagnostics.jl` run at `-t 8` segfaulted after its first chain with the trace
entirely inside Julia's collector: `gc_sweep_page -> gc_sweep_pool_parallel ->
jl_parallel_gc_threadfun`. Not the model. Julia 1.11 runs several GC threads when `-t > 1`,
and the parallel sweep is unstable under heavy multithreaded allocation. The same instability
is the likely cause of the two parallel `-t 4` jobs running 30x slower (GC thrashing rather
than crashing). Fix: `--gcthreads=1`. Applied to the c ladder and to the cluster sbatch.

### The c ladder: prior strength is one barrier, not the barrier (2026-09-11)

100 counties, 4 chains x 6000 sweeps, --gcthreads=1, otherwise identical. 4000-6000 window:

| | c = 100 | c = 30 | c = 10 |
|---|---|---|---|
| mu (cells) | 1.41 | 1.41 | 1.36 |
| theta/s_k aligned | 1.65 | 1.63 | 1.48 |
| s_k*phi aligned | 2.27 | 1.95 | 1.92 |
| FACTOR centre of mass | 3.30 | 3.01 | 3.14 |
| FACTOR activity share | 3.42 | 3.04 | 2.79 |

Weakening c buys ~15% on phi and SATURATES by c = 30; c = 10 adds nothing, and the coarse
summaries stay at ~3. So the hypothesis that the ruggedness comes purely from the phi prior
is wrong: lowering that barrier exposes the next one. The shear penalty at c = 10 (~20 nats)
is still prohibitive, and below c ~ 30 the remaining disagreement is evidently held in place
by something the phi prior was masking -- theta, the D layer, or data-driven modes.

Tempering (`tempered_c.jl`, ladder c = 100 / 20 / 4) is the direct test of whether states
can cross between modes at all. Swap acceptance between adjacent temperatures is the early
signal: near zero between 100 and 20 means the ladder is too coarse.

### Tempering on c: swap acceptance 0.00 / 0.00 (ladder 100 / 20 / 4). Killed.

A phi equilibrated at c = 20 is hundreds of nats worse under c = 100, so adjacent
temperatures never exchange. A workable ladder needs c ratios near 1.2 -- ~20 rungs, 20x
the cost per system. Cluster-scale, not a prototype; not pursued.

### NMF solutions: unique on the subset, not on the full matrix

12 KL-NMF fits of the clipped increments from random starts, factors matched by permutation
on the scaled trajectory:

| | matched cosine to fit 1, min / median | KL loss spread |
|---|---|---|
| N = 100 | 0.960 / 0.999 | 1.0% |
| N = 3104 | 0.887 / 0.931 | 3.9% |

On the subset the DATA has essentially one factorisation, yet MCMC chains from prior draws
disagree at phi Rhat ~2.3. So those modes are PRIOR-INDUCED -- reached by starting far from
the data -- not data-supported. On the full matrix NMF itself is multimodal: competing
factorisations the data genuinely supports, which no sampler cures.

Test in flight: `reported_diagnostics.jl ... 1` (ARGS[11]) initialises each chain from a
different NMF seed. If the chains then agree, the earlier disagreement was an artefact of
prior-draw starts and "chains initialised from the data agree; chains initialised from the
prior do not" is reportable. If they still disagree, the posterior spreads them from a common
start and the factors are not reportable.

### NMF-initialised chains disperse too. This closes the factor question.

| 4000-6000 | prior-draw init | NMF init, one seed per chain |
|---|---|---|
| mu | 1.41/2.03/2.87 | 1.45/2.05/2.65 |
| theta/s_k aligned | 1.65/2.43/3.30 | 1.67/2.34/3.17 |
| s_k*phi aligned | 2.27/3.15/4.02 | 2.37/2.92/3.58 |
| FACTOR centre of mass | 3.30 | 3.20 |
| FACTOR activity share | 3.42 | 3.12 |

Four chains started from four NMF solutions agreeing to cosine 0.999 disperse to the SAME
Rhat as chains from prior draws. The chains do not begin in different modes; they begin
together and the posterior spreads them. The factor decomposition carries genuine posterior
uncertainty along the mixing direction, and the Gibbs sampler explores it too slowly for
chains to agree at any run length tested.

**Twelve interventions, none of which moved factor Rhat by more than ~15%:** eps removal,
eps Metropolis, longer runs (to 15000 on the full matrix), K = 3, sparsity on theta,
per-draw relabelling, tripling N, the scale move, the pairwise shear move (0% acceptance --
the c = 100 prior rejects every mu-preserving mixing by 50-400 nats), the c ladder (saturates
by c = 30), parallel tempering on c (swap acceptance 0.00 -- adjacent rungs never exchange),
and NMF initialisation.

**What is established, and what to write.**
- mu converges on the full matrix (median Rhat 1.22 at 15000 sweeps) and held-out density
  is stable and correctly ordered across every arm. Report those with Rhat / ESS.
- The factorisation is identified only by the c = 100 smoothness prior on phi. That prior
  makes the landscape rugged, and the posterior along the mixing direction is broad enough
  that chains initialised from the data's own NMF solution disperse. Individual wave
  trajectories therefore carry real posterior uncertainty that a heatmap hides.
- Present the factors as exploratory: a single long chain's trajectories with credible bands,
  plus a cross-chain overlay so the reader sees the variability rather than a point estimate.
- On the full matrix NMF itself is multimodal (matched cosine 0.887-0.931, KL loss spread
  3.9%), so this is a property of the data at K = 5, not of the sampler.

NOT recommended: further sampler work. The remaining options (a ~20-rung tempering ladder,
HMC) are either cluster-scale for a diagnostic or contradict the paper's data-augmentation
thesis.

## Sampler-integrity tests (2026-09-12), prompted by the author asking whether the
## disagreement could be a bug rather than the posterior

1. **Determinism.** Two runs, same seed, 8 threads, 80 sweeps: every state block bitwise
   identical. The month-old c = 100 run also reproduces to the digit today. No race.
2. **D pinned at 9, the whole rho / tau / beta / D branch skipped** (the hypothesis being
   that rho -- worst Rhat of any quantity, 6.26 -- and mean D at Rhat 3 were the root):

   | 4000-6000 | D free | D pinned |
   |---|---|---|
   | mu | 1.41/2.03/2.87 | 1.61/2.26/3.11 |
   | theta/s_k aligned | 1.65/2.43/3.30 | 1.53/2.73/3.83 |
   | s_k*phi aligned | 2.27/3.15/4.02 | 2.15/3.11/4.16 |
   | FACTOR centre of mass | 3.30 | 2.87 |

   Unchanged. The non-convergence is in the theta / phi block itself. Remaining suspects
   in that block: the Z-step (latent sums + multinomial thinning), the U update, the V
   update (CRT backward pass + forward sample), the scale move.
3. **Realistic-regime joint test** (K = 5, Q = 25, M = 295, phi in the tens, cumulative
   counts in the thousands, D near the ceiling) -- the default test certifies the kernel
   at K = 2, Q = 3, M = 48, phi ~ 0.01. In flight.
4. **Isolated synthetic recovery** (queued): data generated FROM the model, D and eps pinned
   at their generating values, only theta / phi sampled. If Rhat on mu does not reach ~1 and
   coverage does not reach ~95% with everything else held at truth, the theta / phi block is
   wrong, not slow.

### Results of the integrity tests (2026-09-12, afternoon)

| test | result |
|---|---|
| determinism, 8 threads | bitwise identical |
| D pinned at 9 (rho/tau/beta/D layer removed), real 100-county subset | mu Rhat 1.61 vs 1.41 free -- no help |
| Schein, realistic regime, 3 sweeps | all 17 p > 0.93: NO POWER (kernel barely moves from its own forward draw) |
| Schein, realistic regime, 30 sweeps, paired test | U, V, G, D, eps, beta pass; **tau var / lag1 paired-p = 0.0000** (+1.2%, see below) |
| synthetic from the model, D + eps at truth, K = 3, prior init | mu Rhat 1.45, pooled ESS 30 / 12000, coverage 81%, bias +5% |
| same, **K = 1** (identified up to scale; mu scale-invariant) | **mu Rhat 1.62**, ESS 29, coverage 88% |
| same, K = 1, chains started AT the truth | Rhat 1.55: chains handed the answer still drift apart |
| same, K = 3, started at the truth | Rhat 1.48 |

**Diagnosis.** With K = 1 there is no permutation, no shear, and mu = theta*phi + offset
is invariant to the one remaining (scale) non-identifiability, yet four chains do not agree
on mu after 6000 sweeps and ESS is ~30 of 12000. That is not a rugged landscape; it is a
random walk with an autocorrelation time in the thousands. The mechanism is the
multinomial thinning, not the order-statistic augmentation:

* S | Y, D (the latent sum, ~ D*Y ~ 1e5) is nearly deterministic, so augmenting it loses
  only a factor ~4 of precision on mu. Cheap.
* The thinning then splits S into an offset part (~1e5) and a factor part (~ D*g ~ 75).
  Given that split, theta is known to relative sd 1/sqrt(D*g) ~ 12%; the DATA pin g only
  to sd ~ 1.25*sqrt(mu/D) ~ 80, i.e. ~400%. Fraction of missing information ~ 1 - g/mu,
  DA autocorrelation time ~ 1.5*mu/g sweeps: ~2000 on the synthetic, 1e2 - 1e4 across the
  COVID matrix (mu = cumulative count, g = daily increment).

This is why nothing tried so far helped: NMF init, tempering c, the scale and shear moves,
K, Q, N, hier_b all leave the theta / phi step size untouched. Pinning D did not help for
the same reason. The offset ablation (use_offset = 0) removed most of it (ESS x3) but the
remaining Rhat was the K > 1 factorisation problem, which is separate and secondary.

**Fix (sampler only, model untouched, still data augmentation): collapse the thinning.**
Keep S | Y, D augmented; update theta_{i,k}, phi_{k,t}, eps by univariate slice sampling on
the exact Poisson-with-offset conditional  sum_cells [S log mu - D mu] + prior, with
G = theta*phi maintained incrementally (O(M) per theta, O(N) per phi; ~ 4e7 flops/sweep
on the full matrix, less than the thinning). `collapsed::Bool` field of covid4 (24th),
`collapsed_update!` in models/covid/covid_final.jl. Certification in flight: small-regime
Schein test with `collapsed`, then the K = 1 / K = 3 synthetic recovery.

**tau: a small but real bias in the tau block (paired p = 0.0000 on var and lag1, +1.2%).**
Not the cause of the theta / phi problem (D pinned exonerates the layer) and tau is not
interpreted, but it should be localised. Candidates: the PG draw at |beta'tau| in the
hundreds (Devroye path via rand_pgdevroye), or heavy-tailed replicates under the
inverse-gamma sigma2_tau breaking the paired z; a sign test is in the rerun.

### Collapsed sampler: first result (2026-09-12 evening)

Synthetic from the model, N = 100 x M = 60, K = 1, D and eps at truth, 4 chains x 3000
sweeps (1500 kept):

| sampler | mu Rhat med/p90/max | pooled ESS med/min | coverage |
|---|---|---|---|
| thinning (DA) | 1.62 / 2.15 / 2.63 | 29 / 26 | 88% |
| **collapsed (slice on S-target)** | **1.004 / 1.009 / 1.017** | **1239 / 365** | **95%** |

Same data, same seeds, same S | Y augmentation. ~40-80x the effective sample size from
removing the thinning; nothing else changed. The mechanism in the previous entry is
confirmed. (Note: an earlier batch of "collapsed" synthetic runs was actually thinning --
the flag was not wired through the constructor line; the identical D = 1 outputs exposed
it. Fixed; numbers above are from the corrected script.)

Cost: 0.040 vs 0.012 s/sweep at N = 100, M = 60, K = 3 (2 threads); the slice evaluations
allocate closures and could be tightened, but ESS/second is already ~15x better.

In flight: real 100-county subset with the collapsed sampler (same windows as the c=100
baseline), then Schein certification of the collapsed kernel (small regime, synthreg).

Companion runs, same setup, collapsed sampler (4 chains x 3000, 1500 kept):

| case | mu Rhat med/p90/max | pooled ESS med/min | coverage |
|---|---|---|---|
| K = 1, D == 1 (plain Poisson, no augmentation at all) | 1.004 / 1.007 / 1.012 | 1151 / 733 | 92% |
| K = 3, D free-generated (pinned at truth) | 1.005 / 1.011 / 1.019 | 1528 / 239 | 95% |

K = 3 converging on mu says the K > 1 factorisation multimodality is not binding for mu at
this scale once the thinning is gone. The D == 1 case (pure slice sampler on the exact
posterior) certifies the slice code independently of the order-statistic machinery.

### Is it the temporal prior? (author's recollection: a plain MF mixed fine; the gamma-Markov prior broke it)

Same K = 1 synthetic, phi_{k,t} ~ iid Gamma(start_V1, start_V2) instead of the chain
(`iid_phi` test flag, scale move off), D and eps at truth, 4 x 3000:

| sampler, iid phi prior | mu Rhat med/p90/max | pooled ESS med/min | coverage |
|---|---|---|---|
| thinning | 1.91 / 2.42 / 3.08 | 15 / 13 | 87% |
| collapsed | 1.003 / 1.006 / 1.011 | 1075 / 587 | 95% |

The thinning sampler crawls identically without the temporal prior, and the collapsed
sampler mixes with or without it. The prior is not the cause; the thinning under the
cumulative-count offset is. A plain MF that mixed fine must have been fit without the
Y_{i,t-1} offset in the rate (increments as data), where mu = g and thinning is harmless.

### Can Y_{i,t-1} be dropped instead? (author: keep the standard augmentations, change the model)

`increment_case.jl`, 100-county subset, thinning sampler, SAME 210 held-out events
(20% of counties x last 10 days) scored as log E_post p(Y_{i,t} | Y_{i,t-1}, ...) in both
formulations, 3000 sweeps (2 chains for inferred D):

| data / rate | D = 1 lpd | D inferred lpd (2 chains) | posterior D | gain of D > 1 |
|---|---|---|---|---|
| increments, mu = eps + theta*phi | -9.63 | -13.77 / -9.38 | mean 2.4-3.0; 75% of cells at 1, 24% at Dmax | **-1.94 nats** |
| cumulative, mu = Y_{t-1} + eps + theta*phi | -4.93 | -4.58 / -4.60 | mean 6.3-6.8; 65% at Dmax | **+0.34 nats** |

Without the offset the model loses ~4.7 nats per held-out cell to the cumulative model on
the same events, and inferred D does NOT beat D = 1: 38% of increments are zero, which a
MedPois with mean ~22 and D > 1 cannot produce, so D bimodalises (75% at 1) and the
predictive gets worse. The offset is what makes the under-dispersion pay; dropping it
removes the case study, not just the sampler problem. (Caveat: 3000 sweeps, mu Rhat 1.7
on increments -- the K = 5 factorisation is not identified there either -- but the
held-out gap is 4.7 nats, far outside anything mixing could reverse.)

### Hybrid: collapse ONLY theta, keep the PGDS time-factor augmentation (2026-09-12, late)

The author's constraint: Section 7.2 must show the order-statistic augmentation composing
with the time-factor augmentation (multinomial thinning -> CRT backward pass -> Gamma
forward pass). The fully collapsed sampler discards that. `phi_da = true` keeps the
thinning, the conjugate eps draw and the CRT / forward pass for phi exactly as in the paper
and replaces only the theta Gamma draw with the slice step on the S-target, in the order
z -> eps -> phi -> theta (collapsed step last, z then discarded: a valid partially
collapsed Gibbs sampler; other orders are not). Why it suffices: given phi, the slice step
gives theta its full conditional width so the growth level moves through theta; phi's
shape over time under thinning has conditional width ~1% vs ~3-7% marginal (autocorrelation
~25 sweeps); the theta/phi scale split is the scale move's job.

Same K = 1 / K = 3 synthetic, D and eps at truth, 4 x 3000 (1500 kept):

| sampler | K | mu Rhat med/p90/max | pooled ESS med/min | coverage |
|---|---|---|---|---|
| thinning (paper) | 1 | 1.62 / 2.15 / 2.63 | 29 / 26 | 88% |
| fully collapsed | 1 | 1.004 / 1.009 / 1.017 | 1239 / 365 | 95% |
| **hybrid** | 1 | **1.005 / 1.012 / 1.042** | **1225 / 164** | 93% |
| thinning (paper) | 3 | 1.45 / 1.88 / 2.56 (6000 sweeps) | 30 / 26 | 81% |
| fully collapsed | 3 | 1.005 / 1.011 / 1.019 | 1528 / 239 | 95% |
| **hybrid** | 3 | **1.010 / 1.028 / 1.046** | **1400 / 164** | 95% |

The hybrid converges; its tails are a little heavier than the full collapse (max Rhat
1.04 vs 1.02, min ESS 164 vs 240-365), which is the phi shape's residual thinning
autocorrelation. The paper's composition claim is intact: every augmentation step is
still there, and the change is confined to one conjugate draw. Real-subset run and Schein
certification of the hybrid kernel in flight.

### Hybrid on the real 100-county subset (4 chains x 4000, window 2000-4000) vs thinning

| quantity | thinning Rhat med/p90/max (ESS/chain) | hybrid Rhat med/p90/max (ESS/chain) |
|---|---|---|
| held-out log density | 2.90 (3.1) | 2.09 (4.6) |
| mu (150 cells) | 1.45 / 1.98 / 3.37 (17) | 1.23 / 1.78 / 2.51 (112) |
| theta/s_k aligned | 1.66 / 2.51 / 3.46 (19) | 1.28 / 1.72 / 2.87 (162) |
| s_k*phi aligned | 2.41 / 3.34 / 4.13 (22) | 1.86 / 2.58 / 3.25 (33) |
| FACTOR centre of mass | 3.35 (3.6) | 1.97 (8.3) |
| FACTOR duration | 3.30 (3.1) | 2.24 (5.4) |
| FACTOR activity share | 3.51 (3.1) | 2.43 (13.8) |

Better everywhere (within-chain ESS x6-8 on mu and theta) but NOT converged: chains are
now well mixed locally yet sit in different places (held-out density Rhat 2.1 = they
disagree about the fit itself). Two candidates for the residual: (i) phi's level/shape
under the thinning it still uses (s_k*phi is the worst block, ESS 33); (ii) genuine
multimodality of the K = 5 factorisation on real data (the c = 100 "rugged landscape"
finding). The fully collapsed sampler on the same subset (queued next) separates them:
if it converges, (i); if it plateaus at the same Rhat, (ii). Window 1000-2000 -> 2000-4000
still improving (mu 1.29 -> 1.23, centre of mass 2.39 -> 1.97), unlike the thinning
baseline which had plateaued, so a longer hybrid run is also informative.

Performance note (collapsed sampler, 100-county subset, -t 4): 0.25 s/sweep flat over
2500 sweeps (thinning 0.06, hybrid 0.09); the 4x slowdown seen in one chain of the real
run was transient machine contention (the next chain ran at 0.22 s/sweep throughout).
Profile: the sequential phi slice loop was ~half the wall time (49% thread utilisation),
so it now runs odd columns then even columns in parallel (alternate columns are
conditionally independent given the gamma-Markov neighbours).

### Fully collapsed on the real 100-county subset (4 x 4000, window 2000-4000)

| quantity | thinning | hybrid | fully collapsed |
|---|---|---|---|
| held-out log density | 2.90 (3) | 2.09 (5) | 2.00 (23) |
| mu (150 cells) | 1.45/1.98/3.37 (17) | 1.23/1.78/2.51 (112) | 1.25/1.74/2.44 (120) |
| theta/s_k aligned | 1.66/2.51/3.46 (19) | 1.28/1.72/2.87 (162) | 1.22/1.71/2.48 (150) |
| s_k*phi aligned | 2.41/3.34/4.13 (22) | 1.86/2.58/3.25 (33) | 1.68/2.83/3.33 (23) |
| FACTOR centre of mass | 3.35 (4) | 1.97 (8) | 2.47 (9) |
| FACTOR duration | 3.30 (3) | 2.24 (5) | 2.16 (9) |
| FACTOR activity share | 3.51 (3) | 2.43 (14) | 2.44 (15) |

The full collapse is no better than the hybrid: the residual on real data is NOT phi's
thinning. Chains mix well within themselves and sit in different places. Remaining
candidates: the slow global drift of D (mean D still climbing 4.7 -> 6.1 over 2500
sweeps, 63% at the ceiling, with eps -> 0 alongside; chains at different stages of that
drift disagree on mu), or genuine multimodality of the K = 5 factorisation. Next: the
collapsed sampler with D PINNED at 9 on the same subset -- the earlier D-pinned test was
under thinning, where the theta bottleneck masked everything.

### Synthetic K = 3: the factors themselves (aligned per draw), hybrid vs thinning

| | theta/s_k | s_k*phi | centre of mass | activity share |
|---|---|---|---|---|
| thinning | 1.81 (ESS 16) | 2.80 (13) | 1.51 (18) | 3.50 (13) |
| hybrid | **1.007 (1228)** | 1.44 (22) | 1.36 (31) | 1.50 (20) |

With mu and theta converged, the time factors still do not -- and the reason is visible
in the truth: prior draws of the gamma-Markov chain give near-collinear phi rows
(pairwise cosine 1.00 / 0.92 / 0.91), so the split of activity between two identical
trajectories is a flat direction of the posterior by construction. This is the K > 1
non-identifiability, not a sampler property, and the test as designed cannot tell the
two apart. Rerunning with the truth as K well-separated waves (`waves` flag): if the
factors then converge under the hybrid, the sampler is fine and factor Rhat on real data
measures how separable the real waves are.

### D pinned at 9, collapsed sampler, real 100-county subset (4 x 4000, window 2000-4000)

held-out 1.78 (ESS 64) | mu 1.36/1.74/2.31 (175) | theta/s_k 1.20/1.76/2.23 (178) |
s_k*phi 1.78/2.14/3.43 (63) | centre of mass 2.22 | duration 1.90 | share 2.20.

Unchanged. D's drift is not the residual either. What is left on the real subset: chains
that mix well within themselves (ESS/chain ~175 on mu in a 250-draw window) and sit in
different places with different held-out fit -- distinct local modes of the K = 5
factorisation on 100 counties. Every sampler-side lever has now been pulled (theta,
phi, eps collapsed; D pinned; scale move); the remaining question is whether the modes
are a small-N artefact (K = 5, Q = 25 on 100 counties) that the full matrix resolves.
That is the full-matrix hybrid run on the cluster (COLLAPSED=1). An N = 300 hybrid run is
queued locally overnight as the intermediate check.

### Synthetic K = 3 with SEPARATED true waves (pairwise cosine 0.01-0.12), 4 x 3000

| | mu | theta/s_k | s_k*phi | centre of mass | activity share |
|---|---|---|---|---|---|
| thinning | 1.67 (16) | 1.44 (19) | 2.22 (15) | 1.94 (15) | 2.30 (14) |
| hybrid | **1.009 (1172)** | **1.002 (1930)** | 1.04 / p90 1.41 / max 1.43 (218, min 23) | 1.29 / 1.72 / 1.83 (29) | 1.03 / 1.29 / 1.35 (316) |

With identified factors the hybrid converges on mu, theta and two of the three time
factors; the third (true growth mean is only 8.4 against mu ~ 2e4, so the weakest wave
is barely determined by the data) still has Rhat ~1.4-1.8 on its trajectory and centre
of mass. A 9000-sweep rerun is queued to tell slow-but-converging from a mode. Coverage
of mu on this single dataset is 77% (cells share factors, so one replicate is not a
calibration test; the Schein certification is the correctness check and is next in the
queue). The thinning sampler is at Rhat 1.4-2.3 on everything.

### Schein certification of the hybrid kernel (first pass)

Small regime (N=8, 2000 x 4 sweeps): no paired-p below 0.01; but U_NK.mean paired-p 0.012
(backward 0.988 vs forward 1.007). synthreg (N=100, M=60, K=3, 400 x 30 sweeps): V_KM.mean
paired-p 0.0096 (0.987 vs 1.018), G.mean 0.011, sign tests 0.011-0.014. Each alone is
within multiple testing, but both point the same way (theta / phi scale ~2-3% LOWER after
the kernel), so a small bias in the hybrid cannot be excluded yet. High-power reruns
queued: synthreg 1200 replicates, small 6000 replicates. On paper the partially
collapsed order z -> eps -> phi -> theta(collapsed) is valid (z is redrawn from its full
conditional before anything uses it again); the slice step's target is
a*x - b*e^x + sum_m [S log mu - D mu] with x = log theta.

### Hybrid certification, high power: FAILS in the synthreg regime

1200 replicates x 30 sweeps: theta mean +0.8% (paired p 0.0016), phi mean -2.6% (0.0003),
G mean -1.8% (0.008); small regime (6000 x 4) passes. The thinning theta/phi kernel in
isolation passes synthreg (correct, just slow). The theta slice step is exact in a
quadrature unit test (K = 1 conditional: mean 1.3049 vs 1.3068, sd 0.994 vs 0.990,
quantiles within MC error). So the defect is in the hybrid's composition, not the step.
Localising: (i) full collapse on synthreg, 1200 reps (same theta step, no thinning);
(ii) hybrid with the scale move off. If (i) passes, the partially collapsed ordering
z -> eps -> phi -> theta is at fault in practice despite passing on paper and needs
rethinking (e.g. redraw S,z after theta, or do theta first and phi last).

### Full collapse PASSES synthreg at 1200 replicates; the hybrid's bias is in the phi path

Full collapse, 1200 x 30: no paired-p below 0.05 (theta mean 1.0004 vs 1.0009, G 3.00 vs
3.05 p 0.07). The theta slice step is therefore certified at K = 3, N = 100, threaded.
The hybrid's failure is confined to what it keeps from the thinning sampler: the CRT
backward pass with the le Cam tail approximation (sampleCRTlecam, tol 0.4: only tables
with Bernoulli probability >= 0.4 are exact; the rest are a Poisson with variance
sum p_i instead of sum p_i(1-p_i)). This explains all three certifications at once:
no CRT in the full collapse (passes); the hybrid mixes fast enough that 30 sweeps expose
the approximation (fails); the thinning sampler carries the same approximation but moves
so little in 30 sweeps that the fixed-length Schein test cannot see it ("passes").
GENERAL LESSON: a Schein/Geweke test of fixed length certifies a slow kernel only to the
extent the kernel moves -- it must be run with a fast kernel or many more sweeps.
Test in flight: hybrid with CRT_TOL = 0 (exact CRT), 1200 x 30. If it passes, production
uses a small tolerance (0.02: exact up to i = 49R, tail variance error <= 2%) and the
genes model, which uses the same routine at tol 0.4, needs the same look.

### Confirmed: the le Cam CRT tail approximation was the hybrid's bias

Hybrid, synthreg, 1200 x 30, CRT exact: RESULT consistent (theta mean p 0.018, phi mean
0.068, G mean 0.21, vs 0.0016 / 0.0003 / 0.008 at tol 0.4 -- same forward draws, same
seed, so the change is the kernel's). Production: CRT_TOL = 0.02 (head exact to i = 49R,
tail variance error <= 2%, < 1% of the CRT count's variance on the full matrix). The
scale-move-off run was dropped as moot. The thinning sampler used in the August
full-matrix run carried the 0.4 approximation too; it is slow enough that the effect on
its posterior is small, but the paper's diagnostics should come from the corrected code.

### Real 100-county hybrid rerun with CRT_TOL = 0.02 (4 x 4000, window 2000-4000)

held-out 1.96 | mu 1.21/1.78/2.46 (100) | theta/s_k 1.22/1.69/2.40 (147) | s_k*phi
2.02/2.65/3.53 (27) | centre of mass 2.36 | duration 2.90 | share 2.24. Same as at tol
0.4: the CRT approximation was a bias of the kernel, not the cause of the chains sitting
at different fits. That residual remains the K = 5 factorisation on 100 counties; the
full matrix and the N = 300 run are the tests of whether it is a small-N artefact.

tau bisection (D-layer-only kernel, theta/phi frozen): var +2%, lag1 +2%, paired p ~ 0 --
the bias is inside the D / PG / tau / beta / sigma2 block. Next: naive MvNormal tau draw
vs the Cholesky draw (TAU_NAIVE switch), then single-threaded.

### Separated waves, 9000 sweeps (hybrid): the weak factor is slow, not a mode

mu 1.007/1.015/1.024 (ESS 2123) | theta/s_k 1.001 (5431) | s_k*phi 1.05/1.27/1.28 (251/78) |
centre of mass 1.16/1.41/1.48 (98/51) | share 1.08/1.18/1.20 (194/95). All factor
functionals improved from 3000 sweeps (p90/max 1.44/1.83/1.35 -> 1.28/1.48/1.20).

### tau: the bias reproduces in a textbook toy -- so the TEST is suspect

A 60-line standard Gibbs sampler for the same layer (beta, AR(1) tau, both sigma2, PG
augmentation, MvNormal draws) at N=60, Q=25, M=295 shows var(tau) larger after 30 sweeps
in 62% of 300 pairs (paired z -2.8), the same direction and size as covid_final.jl
(Cholesky, naive, single-threaded all identical). Two reasons the test can reject a
correct kernel here: (1) systematic-scan Gibbs is NOT reversible, so (forward, backward)
is not exchangeable and the sign / paired-symmetry tests have no null guarantee -- only
E[f(fw)] = E[f(bw)] holds; (2) sigma2_tau ~ IG(1,1) gives var(tau) an INFINITE mean, so
the paired mean-difference z is tail-driven. Check in flight: toy with 30 vs 300 sweeps,
log-functionals (finite moments); a correct kernel gives unpaired z ~ 0 and, at 300
sweeps (pair ~independent), frac(fw>bw) ~ 0.5.

### 2026-09-13: the overnight "stall" was an infinite loop, not contention

NMF's multiplicative updates leave exact zeros in W (theta); the slice step started from
log(0) = -Inf and its shrinkage loop spun on NaNs forever (14 h at 100% of one core,
empty log because Julia's file stdout is fully buffered). Fixed: theta / phi slice steps
start from log(1e-8) when the current value is not positive (cannot happen in
stationarity), and the NMF init is floored at 1e-6. Probe after the fix: 0.05 s/sweep.
Lesson recorded in memory: a silent run gets a process sample and a 5-sweep probe, not a
diagnosis by analogy. The NMF-init hybrid, N = 300 hybrid and the 300-sweep tau toy are
running sequentially now.

### NMF-initialised hybrid, real 100-county subset (4 x 4000, window 2000-4000)

held-out 1.96 (23) | mu 1.15/1.56/2.43 (113) | theta/s_k 1.26/1.80/2.95 (154) | s_k*phi
1.91/2.54/3.13 (26) | centre of mass 3.13 | duration 2.63 | share 2.11.
Same as prior init. Chains that start at (near-identical) NMF solutions drift apart while
mixing well locally: the factor posterior on 100 counties has a broad, nearly flat ridge
(factor shapes vary along it, mu only slightly), not an init trap and not sharp modes.
Sampler-side, every local kernel is exhausted; what remains is (i) more counties pinning
the ridge (N = 300 running, full matrix on the cluster), or (ii) reporting the factors
with their honest Rhat as exploratory while mu / held-out carry the diagnostics, or
(iii) a global move along the ridge (likelihood tempering is viable now that local
mixing is fast; the earlier 0% swap rate was under the slow kernel).

### 2026-09-13 16:55: Julia 1.11.2 GC deadlock

N = 300 hybrid run hung 7 min into chain 2 after three healthy chains: 0% CPU, process
sleeping, sample shows jl_gc_wait_for_the_world / jl_safepoint_start_gc -- stop-the-world
GC waiting on a thread that never reaches a safepoint. Same runtime family as the
parallel-GC segfault (worked around with --gcthreads=1). Mitigation: run_wd.sh watchdog
(kill + restart a step whose log stops growing for 15 min). Recommendation: `juliaup
update` to the current 1.11.x (several GC / threading fixes since 1.11.2) before the
cluster campaign; the cluster (Linux) has not shown either symptom.

## Dirichlet theta (2026-09-13)

Author's call: with theta no longer needing conjugacy, put theta_{.,k} on the county
simplex directly -- the interpreted quantity becomes the parameter, the scale lives in
phi ("cases attributable to factor k on day t"), and the scale non-identifiability, the
scale move and the s_k normalisation all go away. Marginally the Gamma model was already
Dirichlet(a) x Gamma(Na, b); what changes is that the gamma-Markov prior now sits on the
scaled trajectory.

* `dir_theta` flag (covid4 field 28). Prior: normalised Gamma(a, 1) columns. Update:
  `theta_simplex!` -- for each factor, two random perfect matchings of the counties per
  sweep; each pair moves mass delta from one county to the other with delta drawn by
  bounded slice sampling from its exact conditional (Dirichlet prior x Poisson S-target on
  the two affected rows). O(M) per move, pairs disjoint so parallel. phi unchanged (PGDS
  pass, hybrid). Scale move disabled.
* c by a scale rule, not a magic number: relative step sd of the chain is 1/sqrt(c phi);
  `c_for_scale(r, phibar) = 1/(r^2 phibar)`. The paper's c = 100 at phi ~ 25 is r ~ 2%;
  the same smoothness at national scale (phibar ~ total daily cases / K) gives c ~ 0.2 on
  the full matrix, ~6 on the 100-county subset. E[phi_1] = phibar via start_V2 = 1/phibar.
* Does NOT address the shear ridge (a column-stochastic mixing preserves the simplex);
  it is a cleanliness / presentation change plus removal of one non-identifiability.
* Certification in flight: Schein small regime (4000 x 4) and dirreg (N=100, M=60, K=3,
  phibar 1000, 400 x 30) with simplex-aware theta functionals (var, mean column max).

**Certified.** Schein small regime (N=8, 4000 x 4): 22 functionals, none below 0.01
(theta var 0.15, max share 0.04). dirreg (N=100, M=60, K=3, phibar=1000, c=2.5, 400 x 30
with the fast hybrid kernel): none below 0.01 (theta var 0.50, max share 0.94, G mean
0.49, phi mean 0.49). First real-subset run in flight (ARG14 = 1 in
reported_diagnostics.jl; DIRICHLET=1 for the cluster, output/covid/diag_dirichlet).

### Dirichlet(1) hybrid, real 100-county subset (c = 5.68 from the scale rule, 4 x 4000, window 2000-4000)

held-out 2.55 (3) | mu 1.26/1.88/2.72 (104) | theta 1.30/1.86/3.19 (136) | phi
2.77/3.41/4.17 (14) | centre of mass 3.71 | duration 2.39 | share 3.38.
WORSE than the Gamma hybrid on the time factors (phi 2.02 -> 2.77, ESS 27 -> 14). Structural:
on the simplex theta cannot carry a factor's scale, so the level of phi_k must move through
the PGDS thinning step, whose conditional given ~1e4-1e5 thinned counts is ~0.5% wide
against a ~7% marginal -- the thinning bottleneck relocated to phi's scale. In the Gamma
model that level moved through theta's slice step (+ the scale move).
Fix keeping the PGDS composition: `phi_scale_move!` -- slice on log kappa_k, phi_k <- kappa
phi_k, target = chain prior of the rescaled row + exact S-target over all cells (O(NM) per
evaluation). Re-certify, rerun the subset.

`phi_scale_move!` added (Dirichlet only): per factor, x = log kappa by slice sampling from
log prior(kappa phi_k) + loglik(kappa) + M x (Liu-Sabatti scale-group move, Haar measure
d kappa / kappa, Jacobian kappa^M), likelihood = exact S-target over all cells, threaded.
Runs after the theta step in both the hybrid and the fully collapsed branch. Dirichlet(0.3)
simplex step certified in the small regime before the move was added (22 functionals,
none low). Re-certification with the move (a = 1 small + dirreg, a = 0.3 small) and the
two subset runs (a = 1, a = 0.3) queued sequentially.

## 2026-09-13 ~19:20: FULL-MATRIX HYBRID RUN SUBMITTED -- SLURM job 1703820 (array 1-4, node r003)

`COLLAPSED=1 sbatch submit_diag.sbatch` from
/net/projects/schein-lab/jimmy/LatentUnderdispersion/revision2_experiments/covid/cluster.
Cluster tree d81f1a4-dirty + rsync of models/covid/covid_final.jl, helper/MatrixMF.jl,
cluster/{run_diag_chain.jl, configs.sh, submit_diag.sbatch, analyze_diag.jl} from the
local tree a47a998-dirty. Gamma theta (a = 1, b = 1), K = 5, Q = 25, Dmax = 9, c = 100,
eps fixed 0.5, hybrid sampler, CRT tol 0.02, scale move on, 4 chains x 15000 sweeps,
16 threads, checkpoint every 500, output/covid/diag_hybrid. Baseline for comparison: the
August thinning run in output/covid/diag (same mask, seeds). Analyse with
`julia --project=../../.. analyze_diag.jl ../../../output/covid/diag_hybrid`.
Author decisions today: keep the Gamma parametrisation (report theta/s_k and s_k phi as
the Dirichlet-Gamma decomposition); Dirichlet code kept as a certified option (dir_theta),
not used. Sparsity (ATHETA=0.3) is the follow-up array if the factors' Rhat stays ~2.

2026-09-14 00:16: all four tasks of 1703820 PREEMPTED on r003 (general partition), requeued
by SLURM, resumed 00:18 from their sweep-6000/6500 checkpoints on h002 / p001 / m002 (~500
sweeps and a JIT warm-up lost per chain). Per-sweep cost before preemption: 3.0 -> 3.9 s
as mean D climbed 5.3 -> 8.3 (four 16-thread tasks sharing one 64-core node). Expect the
12 h wall clock to end each task around sweep 11-12k; one plain resubmit resumes them.

## 2026-09-14: FULL-MATRIX HYBRID RESULT (SLURM 1703820 + resume 1744367; 4 x 15000)

Window 7500-15000, Rhat med/max (ESS/chain med / min / pooled):

| quantity | thinning (Aug) | hybrid |
|---|---|---|
| held-out log density | 2.72 (12/5/17) | 2.96 (12/4/17) |
| mean D | 3.09 (4/4/16) | 2.92 (4/4/16) |
| mu (cells) | 1.22/3.63 (25/5/38) | 1.17/2.39 (886/5/60) |
| theta/s_k aligned | 1.51/3.16 (20/4/25) | 1.40/2.74 (1060/4/34) |
| s_k*phi aligned | 2.34/4.24 (10/4/18) | 2.37/4.10 (11/4/18) |
| FACTOR centre of mass | 2.82/3.51 | 3.46/4.27 |
| FACTOR activity share | 2.96/4.45 | 3.58/3.97 |

Within-chain mixing of mu and the county shares improved ~40x (ESS/chain 25 -> 886,
20 -> 1060); their median Rhat improved modestly. The time factors, mean D and the
held-out density did not: chains settle at different D levels (7.7 vs 8.3), different
fits and different factor shapes -- the same multimodality seen on the 100-county
subset, now confirmed at N = 3104 with a locally well-mixing sampler. Per-sweep cost
2.6 s (quiet node) to 5.7 s (shared node); chain 1 and 3 needed one resume after the
12 h wall clock (preemption at 00:16 cost ~30 min).

### Mode census of the four full-matrix hybrid chains (mode_census.jl, last half)

| chain | held-out ll | mean D | active factors (share > 0.01) | late wave (centre ~270) |
|---|---|---|---|---|
| 4 | -27526 (best) | 8.33 | 5 | yes, share 0.28 |
| 1 | -27935 | 8.19 | 5 | yes, share 0.30 |
| 3 | -29821 | 7.46 | **4** (one factor share 0.00) | no |
| 2 | -30524 (worst) | 6.94 | 5 | yes, share 0.23 |

Growth surfaces agree to corr 0.98-0.997 but differ by rms log-ratio 0.2-0.5; the total
growth trajectory's between-chain sd (1193) exceeds its within-chain sd (822). The modes
differ in the D level (which the factor fit must support: higher D = less noise) and in
whether all five factors are alive: chain 3 lost a factor (a dead factor cannot revive
under the gamma-Markov prior -- phi ~ 0 everywhere, zero thinned counts, prior keeps it
at 0). The best-fitting mode (chains 1 and 4: all factors active, highest D) is the one
the paper's Figure 10 describes. Within-mode diagnostics (chains 1 and 4 only) follow.

Within the best mode (chains 1 and 4 only, window 7500-15000): mu 1.08/1.95 (ESS/chain
968), theta/s_k 1.18/2.02 (1046), s_k*phi 1.93/2.92 (12), centre of mass 2.42, share
2.41, held-out 2.20, mean D 1.90. So even the two chains in the same mode disagree on
the factors: the three large waves match (cosine 0.98 / 0.98 / 0.95, 1.00 for the late
wave), the smallest factor (share ~0.08-0.10) does not (cosine 0.44, centre 159 vs 215) --
the same weakly-identified small factor as the synthetic "weak wave", now on real data.
mu and the county shares are diagnosable; the time factors are reportable as three robust
waves plus one unstable minor factor.

## 2026-09-14 ~17:30: three follow-up full-matrix arrays submitted (hybrid sampler, 4 x 15000 each)

| variant | SLURM | output dir | question |
|---|---|---|---|
| K = 4, Q = 25 | 1744887 | output/covid/diag_hybrid_K4 | does dropping the unstable fifth factor remove the dead-factor / minor-factor modes? |
| K = 5, Q = 1  | 1744891 | output/covid/diag_hybrid_Q1 | is the 25-dim logistic factor model for rho the source of the D-level modes? (per-day rho_t, scalar county loading) |
| K = 5, Q = 5  | 1744895 | output/covid/diag_hybrid_Q5 | the intermediate D layer (Q = 5 matched Q = 25 on held-out density on the subset) |

KFAC / QDIM env knobs in run_diag_chain.jl, directories by variant in configs.sh. Auto-resume
after the 12 h wall clock is handled by the session monitor (resume_<variant>.ids on the
cluster). Reasoning: mean D was the slowest variable of the K = 5 / Q = 25 run (Rhat 2.9,
ESS 4, chains plateauing at different D levels), the ρ layer's τ random walk with learned
σ² is self-sustaining once saturated, and Q = 5 bought no held-out density on the subset.
Sparsity (a < 1) deprioritised: it encourages exactly the dead-factor modes just observed.

## 2026-09-15: Q = 5 (K = 5) full-matrix result (1744895, diag_hybrid_Q5, 4 x 15000)

Window 7500-15000: held-out Rhat 3.30 | mean D 3.75 | mu 1.16/2.19 (ESS 906) | theta/s_k
1.44/3.13 (1061) | s_k*phi 2.04/4.31 | centre of mass 3.54 | share 3.46.
Mode census: chains 1 & 4 (held-out -27623 / -27400, D 8.18 / 8.16) agree closely (cosine
0.96-1.00, rms log-ratio of growth 0.09); chains 2 & 3 each LOST the small first factor
(share 0.00), D 7.3 / 7.9, held-out worse by 400-1800. The D level tracks whether all
factors are alive. So the dimension of the rho layer is NOT the source of the D-level
modes (same pattern at Q = 25 and Q = 5); the recurring mechanism is factor death of the
smallest factor (2 of 4 chains here, 1 of 4 at Q = 25), which drags D and the fit down
with it. Q = 5 costs nothing on fit (best chains' held-out ~ equal to Q = 25's) and can be
adopted for parsimony, but it is not the fix. K = 4 (pending) is the test of the
factor-death mechanism; Q = 1 (pending) closes the Q question at the extreme.

## 2026-09-15: K = 4 and Q = 1 full-matrix results (window 7500-15000, Rhat med/max, ESS/chain)

| | K5 Q25 (paper) | K5 Q5 | **K5 Q1** | K4 Q25 |
|---|---|---|---|---|
| held-out log density | 2.96 (12) | 3.30 (5) | **1.50 (683)** | 2.73 (4) |
| mean D | 2.92 | 3.75 | **1.75** | 2.27 |
| mu (cells) | 1.17/2.39 (886) | 1.16/2.19 (906) | **1.07/2.36 (1120)** | 1.09/2.56 (455) |
| theta/s_k aligned | 1.40/2.74 (1060) | 1.44/3.13 (1061) | **1.20/2.64 (1068)** | 1.49/2.83 (899) |
| s_k*phi aligned | 2.37/4.10 | 2.04/4.31 | 1.81/4.27 | 2.32/4.42 |
| FACTOR centre of mass | 3.46 | 3.54 | 2.46 | 4.20 |
| FACTOR activity share | 3.58 | 3.46 | 2.67 | 3.96 |
| chains with a dead factor | 1 of 4 | 2 of 4 | **0 of 4** | 1 of 4 (chain 4, share 0.01) |
| D level across chains | 6.9-8.3 | 7.3-8.2 | **7.90-7.92** | 8.10-8.26 |
| held-out ll across chains | -30524..-27526 | -29418..-27400 | **-27950..-27810** | -28574..-27790 |

**Q = 1 (per-day rho_t, scalar county loading) removes the D-level modes and the factor
death**: all four chains keep five live factors, agree on D to 0.02, and agree on the fit
to within 140 nats (held-out Rhat 1.50 with 683 effective draws per chain). Chains 1, 3, 4
agree on the factors (cosine 0.92-1.00, growth surfaces within 5-14% rms); chain 2 differs
only on the smallest factor (share ~0.08, cosine 0.26) -- the same weakly identified minor
wave as everywhere else. Fit: -27 810 best vs -27 526 for the best K5/Q25 chain (~300 nats
on 6100 held-out cells), i.e. Q = 1 gives up a little predictive density for a posterior
that is actually sampled. **K = 4 does NOT remove factor death** (chain 4 lost a factor
anyway; centre-of-mass Rhat 4.2): the dying factor is a property of the D layer's
flexibility, not of K. Hypothesis: with a rich rho layer, D can drop to 1 wherever a
factor stops explaining the data, which makes a dead factor "affordable"; with Q = 1 the
per-cell D cannot be tailored, so every factor has to keep earning its keep.
Within-mode (Q = 1, chains 1/3/4): see next block.

Q = 1, chains 1/3/4 only (the three that agree on the minor factor), window 7500-15000:
held-out 1.08 (ESS 922/chain, pooled 78) | mean D 1.37 | mu 1.01/1.70 (1173, pooled 1475)
| theta/s_k 1.03/2.18 (1115, pooled 319) | s_k*phi 1.54/3.55 (14) | centre of mass 2.56 |
share 2.62. So within the mode: mu, the county shares and the held-out density are
converged; the time factors' Rhat is still ~1.5-2.6 because the factor trajectories
have within-chain ESS of only ~14 -- phi's PGDS step remains the slow block (thinning
autocorrelation, expected), and the factor functionals inherit it. That is a run-length /
phi-mixing issue, not multimodality: it would improve with more sweeps or with the fully
collapsed phi step (which gained nothing on the subset but was never tried at full N).

### Q = 1 in detail (2026-09-16)

**Fit vs the paper.** Per held-out cell (6410 cells), last half: Q1 chains -4.338 / -4.360 /
-4.340 / -4.345; the August thinning chains (the paper's fit) -4.346 / -4.771 / -4.578 /
-4.338; Q25 hybrid -4.358 / -4.762 / -4.652 / -4.294. So Q = 1 reproduces the paper's
best-chain fit in EVERY chain (the paper's own chains 2 and 3 were 0.2-0.4 nats/cell worse
-- dead-factor / low-D modes). The 300-nat "cost" quoted earlier was Q25's single best
chain; against the paper's reported fit there is no cost. Against D = 1 the subset gap was
+0.34 nats/cell; a full-matrix D = 1 fit under the same mask is the outstanding number.

**Factors vs Figure 10.** Aligned to August chain 1 (the figure): Q1 chains 1/3/4 match all
five factors, cosine 0.90-1.00; centres 219/154/145/246/272 vs 227/147/136/246/272 days;
durations within a few days; shares 0.09-0.14 / 0.25-0.31 / 0.15-0.16 / 0.21-0.23 /
0.22-0.29 vs 0.17 / 0.22 / 0.14 / 0.17 / 0.30. The only visible change is the minor
factor's share (0.17 -> ~0.10). The August chains disagreed with each other MORE (chains 2
and 3: cosine 0.63 / 0.66 on factor 1; chain 4 lost factor 5) -- the figure was drawn from
one chain of a run whose other chains did not reproduce it; under Q = 1 three of four do.

**Diagnostics.** Converged at 7500-15000: mu 1.07 (ESS/chain 1120), theta/s_k 1.20 (1068),
held-out 1.50 (683), mean D 1.75 (7.90 +- 0.005 within AND across chains). NOT converged:
the time-factor functionals -- per factor, centre Rhat 1.5-4.0, share 1.9-3.8, with
ESS/chain of 4-5 in 1875 stored draws (autocorrelation ~2000 sweeps). Even chains 1/3/4,
which agree on the profiles to cosine 0.9-1.0, give Rhat 1.6-3.8 on the functionals: the
trajectories drift very slowly within a chain, so R-hat has ~2 effective samples per chain
to work with. This is phi's PGDS step (conditional on ~1e5-1e6 thinned counts per (k,t),
0.1-0.3% wide) -- the same thinning autocorrelation as theta had, now on the shape of phi.
Run length cannot fix it (ESS 100 would need ~200k sweeps). Two ways out: (i) report the
factors by cross-chain agreement of their profiles (cosine, centres) rather than Rhat of
functionals with ESS 4; (ii) add group moves on each factor's trajectory against the exact
S-target (time-shift, stretch, tilt exp(b(t - tbar)) -- Liu-Sabatti style, O(NM) per
evaluation, prior ratio from the gamma-Markov chain), which target the centre / duration
/ share directly while keeping the PGDS step. (ii) is a sampler change with the same
composition claim; prototype on the subset within a single chain (ESS of centre of mass
before / after).

### 2026-09-16: alignment was contaminating the theta / phi Rhat

Labels do not switch within chains (Gibbs), but the per-draw Stephens alignment on the
60-county theta profiles is ambiguous between similar factors: on the Q = 1 run chain 3's
assigned permutation flipped among three permutations (933 / 577 / 365 draws) and chain 2's
in 3% of draws; aligning on the K x M phi trajectories instead gives ONE permutation per
chain in 100% of draws. A wrong permutation on a draw inflates Rhat for every theta and phi
element of the two swapped factors. analyze_diag.jl now aligns on phi and also reports the
modal per-chain permutation; the Q = 1 tables are being recomputed. Reporting set fixed by
the author: per-element mu_ij, D_ij, theta_ik, phi_kt only (no functionals). Chain files
now store unthinned draws and full-surface statistics for mu, D and theta (all counties).

Q = 1 re-analysed with phi-based alignment (window 7500-15000; per-chain permutation held
in 100% of draws in every chain): theta/s_k 1.11/2.64 (ESS/chain 1128, pooled 82; was
1.20 / 47 under theta-alignment); s_k*phi 2.05/4.17 (ESS 11/chain, pooled 18; was 1.81 --
the misalignment had been pairing similar factors and hiding part of the disagreement).
So theta improves to ~1.1 and phi is confirmed as the genuinely slow block. Next: the
multiplicative group move on each trajectory (phi_group_move!), tilt + curvature.

### phi group moves: tried, not adopted (2026-09-16)

Multiplicative group moves phi_kt -> phi_kt exp(sum_j a_j g_j(t)) with a_j by slice sampling
on the exact S-target (phi_group_move!, PHI_GROUP switch, off by default). Subset, Q = 1,
4 x 4000, window 2000-4000, s_k*phi aligned Rhat med/p90/max (ESS/chain):
  off                 2.19 / 2.96 / 3.69  (24)
  tilt + curvature    1.85 / 2.86 / 3.74  (21)
  8 cosine functions  1.83 / 2.54 / 3.11  (28)
No material gain: the slow directions of the trajectories are not smooth global
deformations. (First "on" run stalled: Gamma draws with shape ~d underflow tails to 0.0
and the slice sampler started from log density -Inf -- guarded in slice1d, tails floored.)
Decision: report phi per element with its honest Rhat/ESS and, alongside, cross-chain
agreement of the aligned trajectories (Q = 1: 3 of 4 chains at cosine >= 0.9 on every
factor; the 4th differs on the smallest factor only). The sampler for the paper stays:
hybrid (theta collapsed, PGDS phi, CRT tol 0.02, scale move), no group moves.

### 2026-09-16 afternoon: smoothness ladder on the subset (Q = 1, hybrid, 4 x 4000)

Rationale: the failed smooth-deformation move says phi's slow directions are LOCAL
(day-level); the gamma-Markov prior at c = 100 has relative daily step sd 1/sqrt(c phi)
~ 5% (phi ~ 4 per factor-day at any N, since theta has mean 1), i.e. ~40% wander over a
60-day wave, so day-level roughness is essentially unconstrained and each phi_kt is pinned
by its own day's thinned counts. A stiffer c removes the rough modes from the posterior;
too stiff flattens the waves (a rise is ~5%/day for weeks). Ladder c = 300 / 1000 / 3000
vs the c = 100 baseline (ab_q1_off: s_k*phi 2.19/2.96/3.69, ESS 24; held-out to be read
from the per-chain line now printed). Then a D = 1 fit on the subset under the same mask
(the gain over D = 1 for the Q = 1 model, per cell).

### 2026-09-16: D-update speedup ported to covid_final.jl

The author's Sep 15 rewrite of helper/OrderStatsSampling.jl (Poisson pmf recurrence for
os_consts and randTruncDirect) reaches the covid sampler automatically for the latent-sum
step (sampleSumGivenOrderStatistic dispatches on Poisson) -- in every LOCAL run since Sep 15
13:50, but NOT in the cluster runs (the helper was never rsynced; only speed differs). The
covid D update had its own cdf/pdf/logpdf calls; now takes F, f, log f from os_consts
(update_D and the held-out evaluation). Check on 20000 random cells (mu up to 5e5): the
candidate probabilities differ from the incomplete-gamma formulation by at most 8e-7, at
Y ~ mu where the recurrence is the more accurate of the two. Re-certification with the
current helper + this port (Schein synthreg, 1200 reps) and a full-matrix timing are
queued behind the smoothness ladder.

**Smoothness ladder result** (subset, Q = 1, hybrid, 4 x 4000, window 2000-4000; s_k*phi
aligned Rhat med/p90/max, ESS/chain; held-out per cell per chain):
  c = 100    2.19 / 2.96 / 3.69  (24)   ~ -4.58
  c = 300    2.12 / 2.87 / 3.66  (20)
  c = 1000   2.21 / 3.39 / 4.02  (17)   -4.51 .. -4.67
  c = 3000   1.94 / 3.24 / 3.65  (31)   -4.56 .. -4.73
No improvement in phi mixing from a 30x stiffer chain, and the fit spreads slightly across
chains at high c. Together with the failed smooth group moves: the slow directions of phi
are neither the smooth global ones nor removable by a smoothness prior. Closed: keep
c = 100; report phi per element with its Rhat / ESS-per-chain and cross-chain agreement.

**D = 1 baseline on the subset** (same 210 held-out cells, 2 chains): -4.886 / -4.890 per
cell vs ~ -4.58 for the Q = 1 inferred-D model: gain +0.30 nats/cell (Q = 25: +0.34). The
D = 1 fit mixes perfectly with the SAME thinning + PGDS phi step (mu 1.01, theta 1.00,
phi 1.06, phi ESS/chain 54): phi's slow mixing exists only under the inferred-D
likelihood, which is ~5x sharper than Poisson at D ~ 8 -- the sharpness that buys the
information gain is what makes the factor posterior ridged. Not an augmentation defect.

Re-certified (2026-09-16 evening): hybrid + new helper (Poisson recurrence os_consts /
randTruncDirect) + ported D update, Schein synthreg 1200 x 30: consistent (22 functionals,
none below 0.01). Certified code synced to the cluster (helper included this time).

## 2026-09-16 ~15:20: FINAL §7.2 RUN SUBMITTED -- SLURM 1778984 (array 1-4), output/covid/diag_final_Q1

Author decision: adopt K = 5, Q = 1 (per-day rho_t with a scalar county loading), c = 100,
eps fixed 0.5, Dmax = 9. Sampler: hybrid (theta by the collapsed slice step; phi by the
PGDS CRT/forward pass with CRT tol 0.02; scale move; D / PG layer as in the paper), with
the author's Poisson-recurrence helper and the ported D update -- all certified. 4 chains x
15000 sweeps, UNTHINNED, checkpoint every 500; chain files carry mu at 300 cells, D at the
same cells, the 60-county loading profiles, the K x M trajectories, held-out density,
and the full-surface per-segment statistics for mu, D and theta (all 915680 cells /
15520 loadings). Analyse: analyze_diag.jl (per-element Rhat for mu, D, theta, phi; three
windows) and mode_census.jl. Auto-resume after the 12 h wall clock via the session
monitor (resume_final.ids). This is the run the paper's diagnostics table and Figure 10
come from. Still to submit after it (author's call): D = 1 baseline (DFIX=1) and the
held-out rows under the adopted model.

2026-09-17: final run chain 4 done (10.2 h). Chains 1-3 hit the 12 h wall clock at sweeps
8000 / 6500 / 6000; the laptop-side auto-resubmit missed it (machine asleep). Resumed as
1791650 (running on n001 / p001), plus two cluster-side dependent resume blocks 1791654
(afterany:1791650) and 1791655 (afterany:1791654), so completion no longer depends on the
laptop. Tasks whose chain is complete exit immediately.

## 2026-09-17: MID-RUN diagnostics of the final Q = 1 run (chains at 13000/13500/11000/15000)

Computed from checkpoints (ckpt_to_diag.jl -> analyze_diag.jl + mode_census.jl on a compute
node); the chains kept running. Window 5500-11000 (the last common window):

PER-ELEMENT, 300 tracked cells / aligned factors -- Rhat med/max (ESS med/min/pooled):
  held-out log density   1.68        (163/12/66)
  mu (cells)             1.17/2.68   (971/12/159)
  D  (varying cells)     1.01/2.21   (1302/14/1133)   273 of 300 cells vary
  theta/s_k aligned      1.22/2.86   (1016/12/133)
  s_k*phi aligned        2.27/4.31   (19/12/49)
  mean D                 2.08        (18/12/50)

FULL SURFACE, all 915680 cells (split-Rhat from per-segment sufficient statistics):
  mu         median 1.139  p90 1.985  p99 4.734  max 16.99
  D          median 1.010  p90 1.073  p99 1.772  max 320.7 (73796 cells constant, 73795 of
             them identical in every chain -- the Dmax ceiling)
  theta/s_k  median 1.201  p90 2.178  p99 5.074  max 21.18  (all 15520 loadings)

Alignment by phi: the modal permutation holds in 100% of draws in every chain.
Census: held-out per chain -27852 / -27866 / -27900 / -28124; mean D 7.856-7.905 (agreeing
to 0.05); growth surfaces corr 0.985-1.000. Factors: chains 1 and 4 agree on all five
(cosine 0.94-1.00); chains 2 and 3 agree on four and differ on the smallest (share 0.07-0.09,
cosine 0.28 / 0.17) -- chain 3 has that factor at share 0.00, i.e. one dead factor even at
Q = 1 (it did not appear in the Q = 1 exploratory run).
So: D is now the best-behaved quantity (median Rhat 1.01 over the whole surface), mu and
theta are close to converged, phi remains the slow block, and the smallest factor remains
the one the chains disagree about.

### Per-factor diagnostics, final Q = 1 run (mid-run, window 5500-11000, per_factor.jl)

All four chains (one permutation per chain, 100% of draws):
| rank | share | centre | s_k*phi_k Rhat med/max (ESS/chain) | theta_.k Rhat med/max (ESS/chain) | min pairwise cosine |
|---|---|---|---|---|---|
| 1 | 0.29 | 261 | 3.17/4.23 (12) | 1.36/2.86 (22)  | 0.936 |
| 2 | 0.27 | 246 | 2.10/4.31 (13) | 1.22/2.47 (34)  | 0.922 |
| 3 | 0.23 | 156 | 2.23/3.65 (13) | 1.12/2.34 (55)  | 0.937 |
| 4 | 0.14 | 140 | 2.14/3.44 (12) | 1.01/1.78 (636) | 0.925 |
| 5 | 0.06 | 210 | 2.34/3.69 (12) | 1.52/2.37 (19)  | 0.100 |

Chains 1 and 4 only (the two in the best-fitting mode):
| 1 | 0.29 | 270 | 2.02/2.75 (13) | 1.03/1.83 (259)  | 0.999 |
| 2 | 0.25 | 154 | 1.25/3.00 (17) | 1.00/1.26 (1157) | 0.971 |
| 3 | 0.21 | 248 | 1.20/2.90 (17) | 1.01/1.46 (922)  | 0.994 |
| 4 | 0.15 | 144 | 1.34/2.52 (15) | 1.00/1.29 (1135) | 0.996 |
| 5 | 0.09 | 217 | 1.26/2.83 (16) | 1.01/1.30 (853)  | 0.980 |

Within a mode the loadings are converged (theta Rhat ~1.00, ESS 250-1150) and the
trajectories agree (cosine 0.97-0.999) with phi Rhat 1.2-2.0 limited by ESS 13-17 per
chain. Pooling all four chains lifts every factor to Rhat 2-3 because chains 2 and 3 are
in a different mode (chain 3 kills the smallest factor, chain 2 displaces it), which also
shifts the other factors' shares. Recommended reporting: all-four-chains numbers as the
headline plus the mode census (which mode each chain found, with its held-out density --
the chains 1/4 mode fits best: -27852 / -27900 vs -28124), and the within-mode numbers as
the refinement. Rerun on the completed 15000-sweep files when the chains finish.

### 2026-09-17: the local stalls are a Julia 1.11.2 GC deadlock on macOS, not sleep/wake

Both the annealed and the UNANNEALED subset runs froze at 0% CPU mid-run (sweeps ~2000 and
~4000). `sample` shows every thread parked in ijl_gc_collect / jl_safepoint_start_gc: the
stop-the-world GC waiting on a thread that never reaches a safepoint. --gcthreads=1 is
already set and does not prevent it. This is the same signature as the Sep 13 N = 300
stall, which was attributed to the laptop sleeping -- that attribution was wrong; the
machine did not sleep this time. The cluster (Linux, same Julia) has run 15000-sweep
chains without it, so it is macOS-specific. Consequence: local multi-thousand-sweep runs
are unreliable; subset experiments move to the cluster via srun. The annealing A/B was
relaunched there.

### Warm-up annealing of D (author's idea): subset A/B on the cluster, Q = 1, 4 x 6000

Schedule "1:800,3:400,5:400,7:400" -- D pinned at 1, 3, 5, 7 for the first 2000 sweeps,
then inferred. It only shapes the warm-up (those sweeps are burn-in), so the sampler that
produces the reported draws is the certified one and no new validity check is needed.
Window 4000-6000, Rhat med/p90/max (ESS/chain):

| quantity | annealed | plain |
|---|---|---|
| held-out log density | 1.55 (39) | 2.11 (13) |
| mu | 1.17/1.55/2.38 (126) | 1.21/1.67/2.65 (121) |
| theta/s_k | 1.12/1.57/2.47 (170) | 1.29/1.80/2.78 (166) |
| s_k*phi | 1.70/2.28/3.24 (28) | 1.96/2.84/3.31 (29) |

Held-out per chain: annealed -4.556..-4.663 (spread 0.107); plain -4.462..-4.602 (0.139).
Modest but consistent gains on Rhat for held-out density, theta and phi at identical ESS,
i.e. the chains agree better rather than mixing faster -- consistent with annealing
placing them in the same basin. Cheap (the annealed sweeps are faster than ordinary ones,
since D is small during warm-up). Worth adopting for future runs; NOT worth redoing the
finished full-matrix run for on this evidence alone.

Author's correction to the annealing design (2026-09-18): drop the ladder, just hold D = 1
for at least the first half of the warm-up. D barely affects the estimate of mu, and the
intermediate rungs only cost time (the latent-sum draw is O(D): ~1.1 s/sweep at D = 1 vs
~4.5 s at D ~ 8 on the full matrix). Implemented as ANNEAL=n in the cluster driver
(run_diag_chain.jl; its own output directory), and as a schedule string in the subset
driver. The rho layer stays frozen while D is pinned -- conditioning it on D == 1
everywhere would drive rho to zero rather than leaving it uninformative.

## ===== FINAL §7.2 NUMBERS (SLURM 1778984, output/covid/diag_final_Q1) =====
K = 5, Q = 1, c = 100, eps = 0.5, Dmax = 9; hybrid sampler (theta collapsed slice, PGDS phi,
CRT tol 0.02, scale move); 4 chains x 15000 sweeps, unthinned; burn-in 7500, window
7500-15000 (7500 draws/chain). Per-chain wall clock 471-636 min at 16 threads.

PER ELEMENT (300 tracked cells / aligned factors) -- Rhat med/max, ESS med/min/pooled:
  held-out log density  1.41        (140 / 22 / 109)
  mu_it                 1.10/2.43   (1334 / 18 / 289)
  D_it (varying cells)  1.01/2.16   (1457 / 21 / 1532)     267 of 300 cells vary
  theta_ik/s_k          1.16/3.18   (1410 / 16 / 214)
  s_k*phi_kt            2.10/4.20   (26 / 16 / 68)
  mean D                2.29        (24 / 17 / 66)   [global average: posterior sd ~0.003]

FULL SURFACE (every cell; split-Rhat from the per-segment sufficient statistics):
  mu         median 1.102  p90 1.829  p99 4.630  max 16.78   (0 of 915680 constant)
  D          median 1.009  p90 1.064  p99 1.742              (82640 constant, ALL identical
             across chains -- the Dmax ceiling)
  theta/s_k  median 1.149  p90 1.855  p99 5.082              (all 15520 loadings)
Alignment: one permutation per chain, held in 100% of draws in all four chains.

PER FACTOR (window 7500-15000), all four chains:
| share | centre | s_k*phi_k Rhat med/max (ESS/chain) | theta_.k Rhat med/max (ESS/chain) | min cosine |
| 0.28 | 268 | 2.66/3.99 (16) | 1.19/3.18 (44)  | 0.974 |
| 0.27 | 162 | 1.92/3.61 (18) | 1.06/2.08 (120) | 0.943 |
| 0.26 | 245 | 2.03/4.20 (17) | 1.19/2.21 (52)  | 0.962 |
| 0.14 | 143 | 1.96/3.23 (17) | 1.01/1.79 (740) | 0.943 |
| 0.06 | 211 | 2.32/4.00 (17) | 1.52/2.36 (27)  | 0.088 |
Chains 1 and 4 only (same mode): phi Rhat 1.17-2.01, theta 1.00-1.05 (ESS 161-1578),
cosine 0.946-0.999 on every factor including the smallest.

MODE CENSUS: held-out per chain -27852 / -27861 / -27889 / -28044; mean D 7.864-7.912;
growth surfaces corr 0.985-1.000. Chains 1 and 4 agree on all five factors; chain 3 has
the smallest factor dead (share 0.00) and chain 2 displaces it (cosine 0.29).

## 2026-09-18: what the chains actually disagree about, and the K = 3 runs

phi_disagreement.jl on the final run (window 7500-15000), chain-mean trajectories:
| factor (share) | peak day / chain | peak height / chain | mass / chain | mean abs diff |
|---|---|---|---|---|
| 1 (0.28) | 281/281/281/275 | 172k/142k/162k/139k | 5.6/4.2/6.1/4.6 M | 30% of mean height |
| 2 (0.27) | 127/135/127/127 | 68k/61k/65k/68k | 4.5/4.4/5.2/5.6 M | 21% |
| 3 (0.26) | 247/247/247/247 | 106k/134k/103k/109k | 3.8/6.7/4.3/4.2 M | 36% |
| 4 (0.14) | 29/24/29/29 | 32.5k x4 | 3.0/1.9/2.9/2.8 M | 25% |
| 5 (0.06) | 212/71/275/212 | 18k/17k/43k/18k | 1.6/1.3/0.09/1.4 M | 126% |

TIMING is robust (peak days agree to 0-8 days on every substantial factor); AMPLITUDE is
not (masses differ 20-75%), while the total fitted surface agrees (corr 0.985-1.000). The
chains trade mass between temporally OVERLAPPING factors -- 1 with 3 (centres 272/246) and
2 with 4 (150/142). That is the shear direction of the factorisation, not a rugged
landscape: the data pin the waves' position but only weakly pin how the overlap is split.

Author's call: K = 3 (only three factors are reported anyway) provided the fit does not
collapse; no common init, but a better one. Submitted two full-matrix arrays (4 chains x
15000, Q = 1, c = 100, eps 0.5, hybrid), each with a dependent resume block:
  1794432 (+1794437) K = 3, everything else identical to the final run -- the clean
     comparison against K = 5;
  1794433 (+1794438) K = 3 + ANNEAL=3750 (D = 1 for the first 3750 sweeps) + NMFINIT=1
     (each chain starts at its own KL-NMF of the increments; NMF is non-convex so the
     chains stay dispersed -- data-informed, not a common start).
Read-outs: held-out log density per chain vs the K = 5 run's -27852..-28044 (fit), and
phi's per-factor Rhat / cross-chain agreement (identification).

## 2026-09-19: K = 3 with D = 1 warm-up and per-chain NMF init -- the factors agree

output/covid/diag_hybrid_K3_Q1_anneal3750_nmf (1794433/1794438), 4 x 15000, Q = 1, c = 100.

PER FACTOR (window 7500-15000, all four chains, one permutation per chain in 100% of draws):
| share | centre | s_k*phi_k Rhat med/max (ESS/chain) | theta_.k Rhat med/max (ESS/chain) | min cosine |
| 0.44 | 202 | 1.36/1.76 (23) | 1.02/1.52 (650)  | 0.999 |
| 0.41 | 249 | 1.16/1.67 (27) | 1.02/1.61 (313)  | 1.000 |
| 0.15 | 147 | 1.44/2.19 (19) | 1.00/1.41 (1380) | 0.999 |

TRAJECTORY DIFFERENCES between chain means: 2-5% of each factor's mean height (K = 5 was
21-36%, and 126% for its smallest factor). Peak days identical across chains; masses agree
to 1-2%; centres 202 / 249 / 147 in every chain. So at K = 3 the mass-trading between
overlapping factors is gone and the factorisation is reproducible chain to chain.

FIT: held-out log density per chain -28467 / -28539 / -28642 / -30225, against the K = 5
run's -27852 / -27861 / -27889 / -28044. Best-to-best that is ~615 nats over 6410 held-out
cells = 0.10 nats/cell worse. (For scale, inferred D beats D = 1 by ~0.30 nats/cell.)

CAVEAT: chain 1 sits at mean D 5.67 while chains 2-4 are at 7.78-7.80, and its held-out is
1600 nats worse -- the D-level mode has NOT gone away; annealing from D = 1 left one chain
short of the D plateau. Its FACTORS still agree with the others (cosine 1.000), so the
disagreement is confined to the order, not the factorisation. Fix if wanted: a longer
D-inferred settling stretch after the ladder, or a shorter ladder (D climbs from 1 to ~7.8
in about 2000 sweeps once released).
The plain K = 3 run (1794432, no annealing, prior init) is still finishing and will say
how much of this is K = 3 and how much is the new warm-up / init.

## 2026-09-19: K = 3 plain vs K = 3 with the D = 1 warm-up and NMF init

Same data, same sampler, 4 x 15000, Q = 1, c = 100; window 7500-15000.

| | K3 plain | K3 anneal+NMF | K5 (final) |
|---|---|---|---|
| held-out log density Rhat | 1.96 | 2.50 | 1.41 |
| held-out ll per chain | -28447/-28464/-28569/-29804 | -28467/-28539/-28642/-30225 | -27852..-28044 |
| mu (cells) | 1.02/1.72 (1520) | 1.02/1.83 (1494) | 1.10/2.43 (1334) |
| mu, all 915680 cells | 1.019 med, 1.213 p90 | 1.018 med, 1.205 p90 | 1.102 med, 1.829 p90 |
| theta/s_k aligned | 1.02/2.95 (1624) | 1.01/1.61 (1691) | 1.16/3.18 (1410) |
| theta, all 9312 loadings | 1.016 med, 1.298 p90 | 1.009 med, 1.117 p90 | 1.149 med, 1.855 p90 |
| s_k*phi aligned | 1.75/3.49 (28) | 1.30/2.19 (33) | 2.10/4.20 (26) |
| per-factor phi Rhat | 1.63-1.87 | 1.16-1.44 | 1.92-2.66 (+2.32 for the small one) |
| per-factor min cosine | 0.968-0.999 | 0.999-1.000 | 0.943-0.974 (+0.088) |
| trajectory differences | 7-16% | 2-5% | 21-36% (+126%) |

So K = 3 alone gets most of the way (trajectory differences 21-36% -> 7-16%, phi Rhat
2.1 -> 1.75), and the annealed warm-up with the per-chain NMF init closes most of the rest
(-> 2-5%, phi Rhat 1.30, theta Rhat 1.01 with p90 1.12 over all 9312 loadings). Both K = 3
runs put ONE chain at mean D ~5.9 while the other three sit at 7.78-7.81 (held-out 1300
nats worse), so the D-level mode survives both changes -- it is a property of the order
layer, not of K or the init. Fit cost of K = 3: best chain -28447 vs -27852 at K = 5,
i.e. 0.093 nats/cell of the 0.30 nats/cell that inferring D buys over Poisson.
Note the two K = 3 runs find DIFFERENT third factors: plain has centres 249/195/147,
annealed+NMF has 202/249/147 -- the middle factor is the unstable one in both.
