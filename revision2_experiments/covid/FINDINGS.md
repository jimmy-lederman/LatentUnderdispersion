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
