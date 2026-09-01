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
