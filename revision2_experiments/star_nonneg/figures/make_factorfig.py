#!/usr/bin/env python3
"""Build the Section 6.3 factor figure end to end, replacing the hand-assembled
Keynote version.

  panel (a)  the observed count matrix and, beside it, the TRUE factor loadings
  panel (b)  posterior mean loadings for each of the six fitted models, two rows
             of three, including the STAR-NN variant added in revision 2

Everything derives from run output, so the figure cannot drift from the tables:
  data   revison_experiments/factorexperiment/data/CMPfactor.csv
  truth  revison_experiments/factorexperiment/data/CMPU_NK2.csv
  fits   revision2_experiments/star_nonneg/cluster/figsamples/CMP_<model>_<seed>_samples.jld

The .jld files are written by JLD.jl, which is HDF5 underneath, so h5py reads
them directly -- no Julia round trip. Note HDF5 exposes the axes REVERSED
relative to Julia's column-major layout: U is (K, N, draws) here where Julia
sees (draws, N, K).

ALIGNMENT. Factor columns are identified only up to permutation and, under
Gaussian priors, rotation (a sign flip being one special case). Each model's
loadings are therefore matched to the truth before plotting, by exactly the
procedure the reported cosine uses in cluster/cell_lib.jl: permute within each
chain onto that chain's first draw, permute chains onto chain 1, average, then
permute onto the truth, finally matching signs per column. Skipping this would
make the panels incomparable -- the figure would be meaningless rather than
merely unflattering to STAR.

COLOR. The truth tile and every non-negative model share one sequential
white-to-red scale. Gaussian STAR uses a diverging blue-white-red scale centered
at zero because its loadings genuinely go negative; that is the visual point of
the panel, and a sequential scale would conceal it. Each tile is scaled to its
own maximum, since factors are identified only up to a scale exchanged with phi
-- only the PATTERN is comparable across models, never the magnitude.

Usage: python3 make_factorfig.py [seed]
"""
import itertools
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd

# JLD.jl wrote these with compress=true, which is Blosc (HDF5 filter 32001).
# h5py does not bundle that filter, so importing hdf5plugin -- which registers
# it -- must happen BEFORE the first read, or the read fails with a missing
# plugin-directory error rather than anything informative.
try:
    import hdf5plugin  # noqa: F401  (registers the Blosc filter as a side effect)
except ImportError:
    sys.exit("needs hdf5plugin to read the Blosc-compressed .jld files: "
             "pip install hdf5plugin")
import h5py

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DATASET = "CMP"
HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "../../../revison_experiments/factorexperiment/data")
SAMPDIR = os.path.join(HERE, "../cluster/figsamples")

# Panel (b): the six fitted models, two rows of three. STAR-NN is placed next to
# STAR so the only difference between neighbouring tiles is the factor prior.
# (model key, label, Gaussian prior?)
PANELS = [("medpois", "MedPoisson", False),
          ("mednb", "MedNB", False),
          ("poisson", "Poisson", False),
          ("nb", "NB", False),
          ("starnnf_id", "STAR-NN", False),
          ("star_id", "STAR", True)]

SEQ = LinearSegmentedColormap.from_list("wred", ["white", "#C0392B"])
DATA_CMAP = LinearSegmentedColormap.from_list("wdata", ["white", "#C62828"])


def readmat(fname):
    return pd.read_csv(os.path.join(DATADIR, fname)).iloc[:, 1:].to_numpy(float)


def _normcols(A):
    """Column-centered, unit-norm copy, for correlations as a matrix product."""
    B = A - A.mean(axis=0, keepdims=True)
    n = np.linalg.norm(B, axis=0, keepdims=True)
    return B / np.where(n == 0, 1.0, n)


def best_perm(ref, cand, perms):
    """cell_lib.jl's best_perm(Ur, Ut): maximize sum_k |cor(ref_k, cand_p[k])|."""
    S = np.abs(_normcols(ref).T @ _normcols(cand))
    return max(perms, key=lambda p: sum(S[k, p[k]] for k in range(S.shape[0])))


def aligned_mean(path, Utrue):
    with h5py.File(path, "r") as f:
        U = f["U"][:]                       # (K, N, draws)
        nch = int(f["nchains"][()])
        ns = int(f["nsamples"][()])
    K, N, G = U.shape
    U = np.transpose(U, (2, 1, 0)).astype(float)      # -> (draws, N, K)
    perms = list(itertools.permutations(range(K)))

    Ua = np.empty_like(U)
    for ch in range(nch):
        lo = ch * ns
        ref = U[lo]
        for s in range(ns):
            g = lo + s
            Ua[g] = U[g][:, list(best_perm(ref, U[g], perms))]
    ref1 = Ua[0]
    for ch in range(1, nch):
        lo, hi = ch * ns, (ch + 1) * ns
        p = list(best_perm(ref1, Ua[lo:hi].mean(axis=0), perms))
        Ua[lo:hi] = Ua[lo:hi][:, :, p]

    Um = Ua.mean(axis=0)
    Um = Um[:, list(best_perm(Utrue, Um, perms))]
    for k in range(K):                                # sign match (Gaussian only in practice)
        if np.dot(Utrue[:, k], Um[:, k]) < 0:
            Um[:, k] *= -1
    return Um


# ---------------------------------------------------------------- load
Y = readmat(f"{DATASET}factor.csv")
raw = readmat(f"{DATASET}U_NK2.csv")
N = Y.shape[0]
Utrue = raw if raw.shape[0] == N else raw.T

loadings = {"truth": Utrue}
for key, _, _ in PANELS:
    path = os.path.join(SAMPDIR, f"{DATASET}_{key}_{SEED}_samples.jld")
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- rsync the sample file for seed {SEED}")
    loadings[key] = aligned_mean(path, Utrue)
    print(f"aligned {key}")

# ---------------------------------------------------------------- draw
def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def draw_loadings(ax, A, gauss):
    if gauss:
        m = np.abs(A).max()
        ax.imshow(A, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto",
                  interpolation="nearest")
    else:
        ax.imshow(A, cmap=SEQ, vmin=0.0, vmax=A.max(), aspect="auto",
                  interpolation="nearest")
    style(ax)


fig = plt.figure(figsize=(15.5, 4.6))
# Panel (a) holds the data and the true loadings; panel (b) the six fits. The
# left ratio keeps the square 20x20 heatmap close to filling its box -- a wider
# box only pads it with whitespace instead of enlarging it.
outer = fig.add_gridspec(1, 3, width_ratios=[0.58, 0.38, 1.0], wspace=0.10,
                         left=0.03, right=0.985, top=0.90, bottom=0.06)

axa = fig.add_subplot(outer[0, 0])
im = axa.imshow(Y, cmap=DATA_CMAP, aspect="equal", interpolation="nearest")
axa.set_xticks(np.arange(-0.5, Y.shape[1], 1), minor=True)
axa.set_yticks(np.arange(-0.5, N, 1), minor=True)
axa.grid(which="minor", color="0.75", linewidth=0.4)
axa.tick_params(which="both", bottom=False, left=False,
                labelbottom=False, labelleft=False)
style(axa)
fig.colorbar(im, ax=axa, fraction=0.030, pad=0.08)

grid = outer[0, 2].subgridspec(2, 3, wspace=0.18, hspace=0.32)
tiles = []
for i, (key, label, gauss) in enumerate(PANELS):
    ax = fig.add_subplot(grid[i // 3, i % 3])
    draw_loadings(ax, loadings[key], gauss)
    ax.set_title(label, fontsize=13, pad=5)
    tiles.append(ax)

# The truth tile takes its width and height from an ALREADY-PLACED (b) tile, so
# it matches them exactly whatever the grid geometry, and is centered both in
# its own column and vertically in the figure. Sizing it from its own gridspec
# cell would only approximate the match.
tpos = tiles[0].get_position()
tw, th = tpos.width, tpos.height
slot = outer[0, 1].get_position(fig)
# Nudge clear of the colorbar: its tick labels overhang the gridspec box for
# column 0, which matplotlib does not account for, so a tile centered in column
# 1 would otherwise be drawn on top of them.
CBAR_LABEL_PAD = 0.016
axt = fig.add_axes([slot.x0 + (slot.width - tw) / 2 + CBAR_LABEL_PAD,
                    0.5 - th / 2, tw, th])
draw_loadings(axt, Utrue, False)
axt.set_title("Truth", fontsize=13, pad=5)

# Panel labels and the divider sit in FIGURE coordinates, so the figure is saved
# without bbox_inches="tight" -- tight cropping rescales those coordinates after
# the fact and silently moves both. The divider is placed midway between the
# truth tile and the grid, so it tracks any layout change without crowding
# either side.
DIV = (axt.get_position().x1 + tpos.x0) / 2
fig.add_artist(Line2D([DIV, DIV], [0.04, 0.96], color="black", linewidth=3))
fig.text(0.015, 0.07, "(a)", fontsize=20, va="bottom", ha="left")
fig.text(DIV + 0.012, 0.07, "(b)", fontsize=20, va="bottom", ha="left")

out = os.path.join(HERE, "factorsfig")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf")
