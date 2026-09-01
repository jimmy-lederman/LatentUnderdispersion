# Sparse-prior experiment: the cost of losing conjugacy

## Claim

Order statistic models keep exact conjugate updates for **any** Dirichlet
concentration `a` and gamma shape `c`:

    theta_k | counts ~ Dirichlet(a + n_.k)
    phi_kj  | counts ~ Gamma(c + y_.j, d + sum_i theta_ik)

STAR-NN's Gibbs updates are exact **only** at `a = c = 1`. Verified in
`models/STARmodels/revision2/STARMFNN.jl`:

* `update_U!` — the mass-shift conditional on `[0, s]` is proportional to
  `exp(-A t^2 / 2 + B t)`, a truncated normal *only because* Dir(1) is flat on
  the simplex. Under Dir(a) it gains `t^(a-1) (s-t)^(a-1)`.
* `update_V!` — the line `B = B/sigma2 - drate` absorbs the gamma rate into the
  linear term, which works *only because* Gamma(1,d) is an exponential. Under
  Gamma(c,d) it gains `t^(c-1)`.

Sparsity (`a, c < 1`) is the sharp case: both extra factors diverge at the
boundary, which is exactly where a Gaussian proposal has no mass.

## What the claim is NOT

Sampling does not become impossible, and a referee will know it. Exact samplers
exist and should be acknowledged:

* U: propose `t ~ s * Beta(a,a)`, accept w.p. `exp(-A t^2/2 + Bt) / sup`.
* V: propose from the `Gamma(c,d)` prior, accept w.p. the same Gaussian factor
  over `exp(B^2 / 2A)`.

Both are correct; both collapse in acceptance as the likelihood grows
informative (large A). The defensible claim is therefore about COST, not
feasibility: the plug-in conjugate update disappears and every replacement is
paid for in acceptance rate, wall-clock, or mixing.

## DGP (done)

`make_sparse_data.jl` writes `data/SparseCMP{factor,U_NK,V_KM}.csv`.
20x20, K=3, CMP with nu=2, seed 20260812.

Sparsity in U comes from an explicit SUPPORT, not a small Dirichlet
concentration. A concentration below 1 over all 20 rows is degenerate: it puts
60-85% of a factor's mass on a SINGLE row, which is a spike, not a sparse
factor. Instead each factor is active on `SU = 4` rows inside its block, roughly
evenly, and negligible elsewhere. V stays comparatively dense because sparse U
times sparse V leaves mu at zero over 80% of cells -- a degenerate matrix rather
than a sparse one -- and the off-block weight supplies the background level.

Calibrated against the dense Section 6.3 data (`CMPfactor.csv`):

| | dense | sparse |
|---|---|---|
| mu mean | 6.00 | 5.92 |
| Y mean | 1.76 | 1.30 |
| Y zeros | 20.8% | 59.2% |
| U entries < 0.01 | 66.7% | 80.0% |
| largest loading per factor | 0.172 | 0.456 |

Matched on level, genuinely sparser in structure.

## Next: samplers (both, per decision)

1. **Independence MH** — propose from the `a = c = 1` truncated-normal
   conditional already implemented, accept on the ratio of the power factors.
   Acceptance should degrade as `a, c` fall.
2. **Slice sampling** — handles the boundary singularity; costs several density
   evaluations per update.

Report whichever does better, so the comparison cannot be called a strawman.

**Validation gate:** Geweke and Schein tests on each new sampler BEFORE any
production run. An unvalidated update has burned us before.

## Fit grid

`a = c in {1, 0.5, 0.25, 0.1}`, with `a = c = 1` the conjugate baseline.
Models: MedPois, Poisson, STAR-NN, plus STAR-G as a control whose Gaussian
priors do not change at all. ~50 seeds.

## Metrics

Acceptance rate vs `a`; bulk/tail ESS **per second** (the headline -- MH steps
are cheap but mix worse, so wall-clock alone understates the cost); wall-clock
per 1000 iterations; rank-normalized R-hat; info rate; cosine recovery.

## Risk

This can come out the other way: a well-tuned slice sampler may handle `a = 0.1`
fine, in which case the modularity argument rests on implementation effort
rather than performance. Framed as "measure the cost of losing conjugacy" so
either outcome is reportable and we are not fishing.

## Citation

Wallach, Mimno & McCallum, "Rethinking LDA: Why Priors Matter" (2009), for why
the concentration is a substantive modeling choice. VERIFY it supports the
specific claim it is attached to before citing.

---

# Findings so far (samplers built and validated)

`models/STARmodels/revision2/STARMFNNsparse.jl` implements STAR-NN for general
`a` and `c` with three samplers (`:exact`, `:mh`, `:slice`). STARMFNN.jl is
untouched: it is validated and used by the finished Section 6.3 sweep, so it
must stay bit-identical.

## MH acceptance collapses as the prior sharpens

Measured on 20x20, K=3, over 100 sweeps after 50 burn-in:

| a = c | accept (U) | accept (V) |
|---|---|---|
| 1.00 | n/a (exact) | n/a (exact) |
| 0.50 | 0.724 | 0.779 |
| 0.25 | 0.508 | 0.435 |
| 0.10 | 0.207 | 0.139 |

## Cost per effective draw, 1-D conditionals

Thinning needed for KS-indistinguishability from exact rejection draws:

| a = c | MH | slice |
|---|---|---|
| 0.50 | 10-50 | 10 |
| 0.25 | 50-5000 | 10 |

Slice dominates MH by 25-500x at a = 0.25. Independent ESS measurement agrees:
at a = 0.25, MH needs ~150 iterations per effective draw and slice ~30; at
a = 0.10, MH ~290 and slice ~100.

## Two traps hit while validating -- do not repeat

**1. KS needs independent samples.** The first validation pass recorded from
iteration 1 with no burn-in and no thinning, and flagged BOTH samplers as wrong
at every setting. That was an artifact of autocorrelation, not a bug -- the same
trap that produced false Geweke failures earlier in this project. Burn-in and
thinning are built into the test now.

**2. The "exact" reference is itself biased at a = 0.1.** Rejection sampling
proposes from Beta(a,a); at a = 0.1, 1.28% of those draws underflow to exactly
1.0 in Float64 and 3.2% fall below 1e-12. The reference skipped them, so it was
not sampling the target, and the resulting KS failures at a = 0.1 said nothing
about the samplers. Conclusions at a = 0.1 from that test are void. a = 0.25 is
clean (0.003% underflow) and a = 0.5 is exact.

## The structural point this exposes

At a = 0.1 the conditional's mass genuinely concentrates within machine epsilon
of the simplex boundary -- that is a property of the target, not of any sampler.
The order statistic models never meet it: their conditional is
Dirichlet(a + counts), so wherever the data allocate any mass the effective
concentration exceeds 1 and the singularity is regularized away. STAR's
conditional keeps the raw `t^(a-1)` factor no matter how informative the data
are, because the Gaussian likelihood multiplies rather than conjugates. That is
a sharper version of the modularity claim than "STAR needs MH".

## Decisions recorded

* Separate model file, so the production STARMFNN stays bit-identical.
* Both samplers implemented, per instruction; slice is the one to headline,
  since fielding the weaker of the two would be a strawman.
* Validate against an INDEPENDENT exact reference, not the two samplers against
  each other, so a shared bug cannot pass.
* The sparse DGP is built but NOT yet needed for these results, which are
  conditional-level. Whether the full fit experiment uses it is still open.

## Next

Full fit comparison at a = c in {1, 0.5, 0.25}, dropping 0.1 as numerically
degenerate (or including it with the underflow caveat stated). Models: MedPois,
Poisson, STAR-NN (slice), STAR-G. Metrics: bulk/tail ESS per second, wall-clock,
rank-normalized R-hat, info rate, cosine. Geweke/Schein on the full model before
production.

---

# Full-model results (dense CMP, 10 seeds, 4 chains x 2000 after 2000 warmup)

## Decision: run on the DENSE CMP data, not the sparse DGP

A first pass on SparseCMP confounded two effects. STAR fits that data badly
whatever the hyperparameters (info rate -1.73 against MedPois -0.97) because 59%
of entries are zero and the STAR zero bin censors heavily, and the control
STAR-G did not converge at baseline. That mixes a FIT story into what is meant
to be a COMPUTATIONAL one. On the dense CMP data all models are competitive at
a = c = 1 (established by the Section 6.3 sweep), so varying a and c changes one
thing only. The sparse DGP is kept for later use; it is not needed for this
claim.

## Headline: ESS per second

Mean over 10 seeds. Every model is given the same a and c.

| a = c | Poisson | MedPois | STAR-NN slice | STAR-NN MH |
|---|---|---|---|---|
| 1.00 | 757 | 67 | 340 | 341 |
| 0.50 | 625 | 55 | 157 | 119 |
| 0.25 | 492 | 43 | 56 | 16 |

Degradation from a = 1 to a = 0.25, paired within seed:

| model | factor |
|---|---|
| Poisson | 1.5x +/- 0.1 |
| MedPois | 1.6x +/- 0.1 |
| STAR-NN slice | 6.1x +/- 0.2 |
| STAR-NN MH | 21.4x +/- 2.4 |

The sparse posterior is harder for EVERY model -- the conjugate ones lose about
1.5x. Isolating the part attributable to losing conjugacy: 6.1/1.6 ~ 3.8x with
the better sampler, 21.4/1.6 ~ 13x with the natural one.

## MH does not merely slow down, it stops working

Rank-normalized R-hat, mean over seeds, and the fraction of seeds exceeding 1.05:

| a = c | Poisson | MedPois | STAR-NN slice | STAR-NN MH |
|---|---|---|---|---|
| 1.00 | 1.008 (0%) | 1.016 (0%) | 1.008 (0%) | 1.008 (0%) |
| 0.50 | 1.010 (0%) | 1.024 (0%) | 1.020 (0%) | 1.073 (30%) |
| 0.25 | 1.018 (0%) | 1.051 (60%) | 1.044 (20%) | 1.821 (100%) |

Independence MH fails to converge on EVERY seed at a = 0.25. Slice stays valid
(mean 1.044) and is the sampler to report; fielding MH as the implementation
would be a strawman, which is why both were built.

## Honest caveat

MedPois exceeds R-hat 1.05 on 60% of seeds at a = 0.25 -- worse than STAR-NN
slice at 20%. Sparse hyperparameters make the posterior harder for the order
statistic models too. What they do NOT pay is the ESS/s penalty: 1.6x against
6.1x. The claim is therefore about the cost of the UPDATE, not about the order
statistic models being immune to sparse priors.

## Validation

* 1-D conditionals vs exact rejection draws: `validate_samplers.jl`
* Whole model, Schein joint test: `validate_schein_sparse.jl` -- 5/5 pass,
  including both samplers at a = c = 0.5 and 0.25
* MH and slice are bit-identical at a = c = 1 (both branch to the exact draw),
  which the results confirm

## Known limitation in this script

`run_sparse_cells.jl` computes cosine from a plain posterior mean with no
per-draw label-switch alignment, so its cosine column is unreliable and swings
between seeds. ESS/s and R-hat are the metrics this experiment is for. Use
`cluster/recompute_cosine.jl` if recovery is wanted here.

---

# Pushing a and c below 0.25

## A real bug the extension exposed

At a = 0.01 the MH branch died with `log(-1.79e-14)`. `truncgauss` can return a
value a rounding error outside [0, s], and the power-factor ratio takes the log
of `s - prop`. Fixed with a clamp to [4*eps(s), s - 4*eps(s)], skipping the move
when the interval is narrower than that. The V branch got the analogous guard on
its lower end.

The clamp is not cosmetic: below roughly a = 0.05 the conditional's mass
genuinely lies within machine epsilon of the simplex boundary, so Float64 cannot
represent it faithfully however the update is written. That is a property of the
TARGET, not of the sampler.

## MH becomes wrong, not merely slow

Schein joint test, small model, 4000 replicates at nthin = 20:

| a = c | MH | slice |
|---|---|---|
| 0.50 | PASS (p = 0.58) | PASS (p = 0.31) |
| 0.25 | PASS (p = 0.29) | PASS (p = 0.45) |
| 0.10 | PASS (p = 0.10) | PASS (p = 0.14) |
| 0.05 | PASS (p = 0.44) | PASS (p = 0.61) |
| 0.01 | **FAIL** (Bonferroni 0.010; var(U_NK) p = 0.001) | PASS (p = 0.28) |

MH is valid down to a = 0.05 and invalid at 0.01. That matches its behaviour:
at a = 0.01 the MH chain wanders to U entries of 1e-163, where the clamp starts
to bias it, while slice stays near 1e-16.

Strictly the failure is of "MH as implementable in Float64" rather than of MH in
exact arithmetic -- but since the clamp was required to stop it crashing, that
distinction does not help a practitioner.

## Consequence for what gets reported

Slice is the sampler to headline: valid at every setting tested, 0.01 included.
MH is reported as the natural implementation that is NOT good enough -- it costs
21x at a = 0.25 and fails validation at 0.01. Fielding only MH would be a
strawman; fielding only slice would hide that the obvious implementation breaks.

## Longer run

The 2000-draw results are kept in `results_short2000/`. The production grid is
a = c in {1, 0.5, 0.25, 0.1, 0.05, 0.01} x {poisson, medpois, starnn_slice,
starnn_mh} x 10 seeds, at 4 chains x 6000 draws after 6000 warmup -- 3x the
earlier budget, so the models being compared have converged rather than being
compared mid-transient.

---

# Extended grid, long run: 240 cells, 10 seeds, 4 chains x 6000 after 6000 warmup

## Where the comparison is valid at all

Fraction of seeds with rank-normalized R-hat > 1.05:

| a = c | Poisson | MedPois | STAR-NN slice | STAR-NN MH |
|---|---|---|---|---|
| 1.00 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.50 | 0.0 | 0.0 | 0.0 | 0.1 |
| 0.25 | 0.0 | 0.0 | 0.0 | **1.0** |
| 0.10 | 0.0 | 0.6 | 0.2 | 1.0 |
| 0.05 | 0.7 | 1.0 | 1.0 | 1.0 |
| 0.01 | 1.0 | 1.0 | 1.0 | 1.0 |

**Below a = 0.05 NOTHING converges, conjugate or not.** Poisson's mean R-hat is
1.86 at a = 0.01 and MedPois's is 1.98. Tripling the budget did not fix it. At
those settings the posterior itself is hard -- the near-degenerate simplex
geometry, not the update rule -- so results there say nothing about conjugacy
and must not be used for the claim. STAR-NN MH's R-hat at a = 0.01 is 1.2e14,
which is total breakdown and matches its Schein FAIL.

The clean window is **a = c in {1, 0.5, 0.25}**, where Poisson, MedPois and
STAR-NN slice all converge on every seed.

## Cost inside the clean window

ESS/s relative to a = c = 1, paired within seed:

| model | 0.50 | 0.25 |
|---|---|---|
| Poisson | 1.3x | 1.9x |
| MedPois | 1.3x | 1.8x |
| STAR-NN slice | 2.3x | 6.6x |
| STAR-NN MH | 3.1x | 23.1x (does not converge) |

The conjugate models lose about 1.9x at a = 0.25 simply because the posterior is
harder. STAR-NN slice loses 6.6x, so the part attributable to losing conjugacy
is roughly **3.5x**. MH loses 23x and fails to converge on every seed.

MH acceptance over the grid: 0.75, 0.52, 0.27, 0.17, 0.12 at a = 0.5 down to
0.01.

## A limitation to state plainly

On the DENSE CMP data a sparse prior is misspecified, and it shows: MedPois's
info rate degrades monotonically from -1.488 at a = 1 to -1.632 at a = 0.01. A
referee can fairly ask why anyone would impose a prior that hurts.

Answering that needs the sparse DGP after all -- to show a regime where a < 1
IMPROVES fit, so that the computational cost is a cost worth caring about. The
dense data isolates the computational effect cleanly, which is why the headline
numbers come from it; the sparse data is what motivates wanting the prior. Both
are therefore probably needed, contrary to the earlier decision to shelve the
sparse DGP.

## Decisions recorded

* Clean window is a >= 0.25; below 0.05 no model converges and no conjugacy
  claim can be made from that region.
* Slice headlines; MH is reported as the natural implementation that fails,
  both on cost (23x) and on validity (Schein FAIL at 0.01).
* The 2000-draw run is archived in `results_short2000/`; the 6000-draw run in
  `results/` supersedes it.
