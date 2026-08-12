#!/usr/bin/env python3
"""Generate the Section 6.3 LaTeX tables directly from the per-cell experiment
output. Nothing is transcribed by hand, so the paper cannot drift from the runs.

Inputs (per-cell CSVs, one row each, 3200 cells = 4 datasets x 8 models x 100 seeds):
  results/       info rate, timing, and the sweep's own ESS/Rhat   (run_task.jl)
  results_diag/  bulk/tail ESS and rank-normalized Rhat            (recompute_diag.jl)
  results_cos/   factor recovery under both invariance groups      (recompute_cosine.jl)

Outputs (tabular environments only -- wrap \\caption/\\label yourself):
  tables/inforate.tex     Table 1
  tables/diagnostics.tex  Appendix D, one tabular per dataset
  tables/recovery.tex     factor recovery, permutation vs rotation alignment

Usage:  python3 make_tables.py [--bold {paired,se}] [--tol T]

BOLDING. Referee 1 point (d) objected that the old text read an ordering off
differences well inside the noise, so the rule is explicit and stated in the
caption. Every model sees the SAME heldout mask at a given seed, so model
differences are PAIRED and the paired interval is much tighter than the
unpaired one -- on NB data MedNB sits 0.004 +/- 0.002 behind NB, resolvable
only because of the pairing.

  --bold paired (default)  bold the best, plus any model whose paired 95%
                           interval against the best covers 0, i.e. genuinely
                           indistinguishable given 100 replicates.
  --bold se                bold the best, plus any model whose unpaired
                           mean +/- 1.96 SE interval overlaps the best's. This
                           is what the submitted table effectively did; it is
                           more generous and ignores the pairing.
  --tol T                  additionally bold anything within T nats of the best
                           regardless of significance. Use when a difference is
                           statistically resolvable but scientifically nil
                           (T=0.01 bolds MedNB alongside NB on NB data). Default
                           0.0 = off.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")

# columns ordered by dispersion, matching the submitted Table 1
DATASETS = [("GC", r"GC $(\mathbb{D}[Y] \approx .33)$"),
            ("CMP", r"CMP $(\mathbb{D}[Y] \approx .5)$"),
            ("Poisson", r"Poisson $(\mathbb{D}[Y] = 1)$"),
            ("NB", r"NB $(\mathbb{D}[Y] = 4)$")]

MODELS = [("poisson", "Poisson"),
          ("medpois", "MedPois"),
          ("nb", "NB"),
          ("mednb", "MedNB"),
          ("star_id", r"STAR \,\,$g(x)=x$"),
          ("star_sqrt", r"STAR \,\,$g(x)=\sqrt{x}$"),
          ("starnnf_id", r"STAR-NN \,\,$g(x)=x$"),
          ("starnnf_sqrt", r"STAR-NN \,\,$g(x)=\sqrt{x}$")]


def load(subdir):
    files = glob.glob(os.path.join(HERE, subdir, "*.csv"))
    if not files:
        sys.exit(f"no CSVs in {subdir}/ -- run the sweep first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def check_complete(df, name):
    n = df.groupby(["dataset", "model"]).size()
    if n.min() != n.max():
        print(f"WARNING {name}: uneven replicates, {n.min()}-{n.max()} per cell",
              file=sys.stderr)
    return n.min()


def bold_set(sub, rule, tol):
    """Which models to bold in one dataset column. sub is indexed by seed."""
    means = sub.mean()
    best = means.idxmax()                       # info rate: higher is better
    keep = {best}
    for m in means.index:
        if m == best:
            continue
        if tol > 0 and (means[best] - means[m]) <= tol:
            keep.add(m)
            continue
        if rule == "paired":
            d = (sub[m] - sub[best]).dropna()
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            if abs(d.mean()) <= half:           # interval covers 0
                keep.add(m)
        else:                                   # unpaired interval overlap
            n_m, n_b = sub[m].count(), sub[best].count()
            h_m = 1.96 * sub[m].std(ddof=1) / np.sqrt(n_m)
            h_b = 1.96 * sub[best].std(ddof=1) / np.sqrt(n_b)
            if means[m] + h_m >= means[best] - h_b:
                keep.add(m)
    return keep


def inforate_table(df, rule, tol):
    piv = df.pivot_table(index=["dataset", "seed"], columns="model",
                         values="inforate")
    bolds, cells = {}, {}
    for ds, _ in DATASETS:
        sub = piv.loc[ds]
        bolds[ds] = bold_set(sub, rule, tol)
        for key, _ in MODELS:
            v = sub[key]
            cells[(ds, key)] = (v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.count()))

    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"info rate ($\uparrow$)~\cref{eq:inforate} & "
         + " & ".join(h for _, h in DATASETS) + r" \\", r"\midrule"]
    width = max(len(lbl) for _, lbl in MODELS)
    for key, lbl in MODELS:
        row = []
        for ds, _ in DATASETS:
            mu, se = cells[(ds, key)]
            s = f"{mu:.2f} $\\pm$ {se:.2f}"
            row.append(r"\textbf{" + s + "}" if key in bolds[ds] else s)
        L.append(f"{lbl:<{width}} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def diagnostics_table(diag, base):
    m = diag.merge(base[["dataset", "model", "seed", "time_s"]],
                   on=["dataset", "model", "seed"])
    out = []
    for ds, head in DATASETS:
        s = m[m.dataset == ds]
        out.append(f"% ---- {ds} ----")
        out += [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                r" & \multicolumn{3}{c}{fitted mean $\mu_{ij}$} & "
                r"\multicolumn{3}{c}{heldout log-density $\ell$} & \\",
                r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                r"model & ESS$_{\mathrm{bulk}}$ & ESS$_{\mathrm{tail}}$ & "
                r"$\widehat{R}_{\max}$ & ESS$_{\mathrm{bulk}}$ & "
                r"ESS$_{\mathrm{tail}}$ & $\widehat{R}_{\max}$ & s/1k iter \\",
                r"\midrule"]
        for key, lbl in MODELS:
            g = s[s.model == key]
            out.append(
                f"{lbl} & {g.essmu_bulk_med.mean():.0f} & "
                f"{g.essmu_tail_med.mean():.0f} & {g.rhatmu_rank_max.max():.3f} & "
                f"{g.essll_bulk.mean():.0f} & {g.essll_tail.mean():.0f} & "
                f"{g.rhatll_rank.max():.3f} & {(g.time_s / 40).mean():.2f}" + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def recovery_table(cos):
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"cosine to truth ($\uparrow$) & "
         + " & ".join(h for _, h in DATASETS) + r" \\",
         r"\midrule"]
    for key, lbl in MODELS:
        row = []
        for ds, _ in DATASETS:
            g = cos[(cos.dataset == ds) & (cos.model == key)]
            row.append(f"{g.cos_perm.mean():.2f} / {g.cos_proc.mean():.2f}")
        L.append(f"{lbl} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", choices=["paired", "se"], default="paired")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    base = load("results")
    nrep = check_complete(base, "results")
    print(f"{len(base)} cells, {nrep} replicates per (dataset, model); "
          f"bold rule = {a.bold}, tol = {a.tol}", file=sys.stderr)

    tables = {"inforate.tex": inforate_table(base, a.bold, a.tol)}
    for sub, fn, key in [("results_diag", diagnostics_table, "diagnostics.tex"),
                         ("results_cos", recovery_table, "recovery.tex")]:
        if glob.glob(os.path.join(HERE, sub, "*.csv")):
            d = load(sub)
            check_complete(d, sub)
            tables[key] = fn(d, base) if key == "diagnostics.tex" else fn(d)
        else:
            print(f"skipping {key}: no {sub}/", file=sys.stderr)

    for name, body in tables.items():
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(body + "\n")
        print(f"\n%%%%%%%% tables/{name} %%%%%%%%\n{body}")


if __name__ == "__main__":
    main()
