#!/usr/bin/env python3
"""Aggregate per-cell CSVs from results/ into the paper tables:
info rate (mean +/- 1.96 SE), diagnostics, ESS/s, and cosine-to-truth."""
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(HERE, "results", "*.csv"))
if not files:
    raise SystemExit("no result files found")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"{len(df)} cells collected "
      f"({df.dataset.nunique()} datasets x {df.model.nunique()} models, "
      f"seeds per cell: {df.groupby(['dataset','model']).size().min()}"
      f"-{df.groupby(['dataset','model']).size().max()})\n")

order = ["poisson", "medpois", "nb", "mednb", "star_id", "star_sqrt",
         "starnnf_id", "starnnf_sqrt"]

def agg(g):
    n = len(g)
    return pd.Series({
        "inforate": f"{g.inforate.mean():.3f} ± {1.96*g.inforate.std(ddof=1)/np.sqrt(n):.3f}",
        "essmu_med": f"{g.essmu_med.mean():.0f}",
        "essmu_min": f"{g.essmu_min.mean():.0f}",
        "rhatmu_max": f"{g.rhatmu_max.mean():.3f}",
        "essll": f"{g.essll.mean():.0f}",
        "rhatll": f"{g.rhatll.mean():.3f}",
        "ESS_per_s": f"{(g.essmu_med/g.time_s).mean():.0f}",
        "s_per_fit": f"{g.time_s.mean():.0f}",
        "cosine": "—" if g.cosine_to_truth.isna().all() else f"{g.cosine_to_truth.mean():.3f}",
    })

for ds, sub in df.groupby("dataset"):
    print(f"### {ds}")
    tab = sub.groupby("model").apply(agg, include_groups=False)
    tab = tab.reindex([m for m in order if m in tab.index])
    print(tab.to_string(), "\n")
