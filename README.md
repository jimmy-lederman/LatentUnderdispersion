# Modeling Latent Underdispersion with Discrete Order Statistics

We provide code for the main data augmentation algorithm as well as code and data to reproduce each case study in the paper.

## Repository Structure

```
├── analysis/       # Scripts to run models on each case study
├── models/         # Model definitions
├── helper/         # Core sampling and data augmentation utilities
├── data/           # Datasets for the 4 case studies
```

## Data

Each subfolder in `data/` contains a `data_dictionary.txt` describing all files and columns.

| Folder | Case Study | Description |
|--------|-----------|-------------|
| `data/birds/` | Finnish bird abundance | 2826 survey sites × 137 species, with covariates |
| `data/covid/` | COVID-19 case counts | 3105 US counties × 296 days, cumulative counts |
| `data/flights/` | Frontier Airlines flights | 42773 flights with air time, distance, and airport metadata |
| `data/genes/` | RNA-seq gene expression | 1000 tissue samples × 309 genes from TCGA |

## Models

Each model is defined as a Julia struct extending the `MatrixMF` base class from `helper/MatrixMF.jl`.

| Folder | Case Study | Description |
|--------|-----------|-------------|
| `models/birds/` | Birds | Poisson order statistic model with and without covariates |
| `models/covid/` | COVID-19 | Poisson order statistic model for cumulative counts; includes `ablation/` subfolder with ablation study variants |
| `models/flights/` | Flights | Poisson order statistic model for air travel times |
| `models/genes/` | Genes | Negative binomial order statistic model for RNA-seq counts |

## Helper Code

| File | Description |
|------|-------------|
| `helper/MatrixMF.jl` | Base class for all models; defines the MCMC sampling interface |
| `helper/OrderStatsSampling.jl` | Conditional order statistic sampling (Algorithm B from the paper) |
| `helper/PoissonOrderPMF.jl` | PMF computation for Poisson order statistics |
| `helper/NegBinPMF.jl` | PMF computation for negative binomial order statistics |

## Analysis Scripts

Scripts in `analysis/` run the models on full or held-out data for each case study:

- `analysis/birds/` — full model run (covariates and no covariates), held-out evaluation (no covariates)
- `analysis/covid/` — full model run, held-out evaluation
- `analysis/flights/` — held-out evaluation
- `analysis/genes/` — full model run

All scripts take command-line arguments in a standard order:

- **Full-data scripts:** `chainSeed D K [Q]`
- **Held-out scripts:** `maskSeed chainSeed D K [Q] [type]`
- **Flights held-out:** `maskSeed chainSeed D type g`

`K` is the latent dimension for the mean. `Q` is the latent dimension for the order D. `type` selects the model variant (see each script for details). `g` controls which transformation to use for the STAR version (flights only).

## Getting Started

1. Install [Julia](https://julialang.org/) (developed with Julia 1.x).

2. Install dependencies:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. Run an analysis script, e.g.:
   ```bash
   julia --project=. analysis/flights/runflights_heldout.jl 101 101 3 1 0
   ```

Output is saved to `output/`.
