# Modeling Latent Underdispersion with Discrete Order Statistics

Code and data accompanying the paper on discrete order statistic models for latent underdispersion.

## Repository Structure

```
├── analysis/       # Scripts to run models on each case study
├── models/         # Model definitions (Julia)
├── helper/         # Core sampling and PMF utilities
├── data/           # Datasets for the 4 case studies
```

## Data

Each subfolder in `data/` contains a `data_dictionary.txt` describing all files and columns.

| Folder | Case Study | Description |
|--------|-----------|-------------|
| `data/birds/` | Finnish bird abundance | 2826 survey sites × 137 species, with land-cover covariates |
| `data/covid/` | COVID-19 case counts | 3105 US counties × 299 time periods, cumulative counts |
| `data/flights/` | Frontier Airlines flights | 42773 flights with air time, distance, and airport metadata |
| `data/genes/` | RNA-seq gene expression | 1000 tissue samples × 309 genes from TCGA |

## Models

Each model is defined as a Julia struct extending the `MatrixMF` base class from `helper/MatrixMF.jl`.

| Folder | Case Study | Description |
|--------|-----------|-------------|
| `models/birds/` | Birds | Poisson order statistic model with and without covariates |
| `models/covid/` | COVID-19 | Negative binomial order statistic model for cumulative counts; includes `ablation/` subfolder with ablation study variants |
| `models/flights/` | Flights | Poisson order statistic model for air travel times |
| `models/genes/` | Genes | Negative binomial order statistic model with Polya-Gamma augmentation for RNA-seq counts |

## Helper Code

| File | Description |
|------|-------------|
| `helper/MatrixMF.jl` | Base class for all matrix factorization models; defines the MCMC sampling interface |
| `helper/OrderStatsSampling.jl` | Conditional order statistic sampling (Algorithm B from the paper) |
| `helper/OrderStatsSampling_fast.jl` | Optimized version of Algorithm B |
| `helper/PoissonOrderPMF.jl` | PMF computation for Poisson order statistics |
| `helper/NegBinPMF.jl` | PMF computation for negative binomial order statistics |

## Analysis Scripts

Scripts in `analysis/` run the models on full or held-out data for each case study:

- `analysis/birds/` — full covariate model, full D model, held-out D evaluation
- `analysis/covid/` — full model run, held-out evaluation
- `analysis/flights/` — held-out evaluation
- `analysis/genes/` — full model run, held-out evaluation

## Requirements

- [Julia](https://julialang.org/) (developed with Julia 1.x)
- Julia packages: `Distributions`, `LinearAlgebra`, `LogExpFunctions`, `SpecialFunctions`
