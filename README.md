# Empirical-Asset-Pricing-via-Machine-Learning
### Partial Replication of Gu, Kelly & Xiu (2020)

> **Paper:** Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223–2273.

---

## Overview

This project partially replicates Gu et al. (2020), one of the most influential papers in quantitative finance, which demonstrates that machine learning models can meaningfully predict cross-sectional stock returns using firm characteristics.

Due to computational constraints, this replication uses **36 firm characteristics** (vs. 94 in the original paper) and covers the period **2010–2021** using CRSP/Compustat data accessed via WRDS.

Four models are implemented and compared:

| Model | Type | Key Hyperparameters |
|---|---|---|
| Elastic Net | Regularized linear | α, l1_ratio (tuned via TimeSeriesSplit CV) |
| PCR | Dimensionality reduction + linear | n_components (tuned on validation set) |
| Random Forest | Ensemble / non-linear | n_estimators=100, max_depth=20 |
| Neural Network (NN1) | Deep learning | 3 hidden layers: 32-16-8, ReLU, Adam |

---

## Key Results

Models are evaluated using **out-of-sample R²** (R²_OOS), defined as:

$$R^2_{OOS} = 1 - \frac{MSE_{model}}{MSE_{benchmark}}$$

where the benchmark is the historical mean return predictor (Campbell & Thompson, 2008).

### Main Period (Train: 2010–2017 | Val: 2018–2019 | Test: 2020–2021)

| Model | Val R²_OOS | Test R²_OOS |
|---|---|---|
| Elastic Net | +0.00374 | -0.00162 |
| PCR | +0.00342 | -0.00203 |
| Random Forest | +0.00338 | -0.00913 |
| Neural Network | +0.00678 | -0.00466 |

### Robustness Check: Pre-COVID vs COVID Period

| Model | Pre-COVID R²_OOS (2018) | COVID-period R²_OOS (2020–21) |
|---|---|---|
| Elastic Net | +0.00000 | -0.00162 |
| PCR | +0.00731 | -0.00203 |
| Random Forest | +0.01095 | -0.00913 |
| Neural Network | +0.00475 | -0.00466 |

> **Average Pre-COVID R²_OOS: +0.0058 | Average COVID R²_OOS: −0.0044**
> MSE increased 101.5% from pre-COVID to COVID period, confirming significant regime shift.

---

## Methodology

### Preprocessing Pipeline
Following Gu et al. (2020) exactly:

1. **Feature selection** — 36 characteristics with <30% missing values (2010–2021), prioritised by importance from Figure 5 of the original paper
2. **Cross-sectional median imputation** — missing values filled with the cross-sectional median for that month
3. **Winsorization** — extreme values clipped at 1st/99th percentiles per feature
4. **Rank transformation** — all features scaled to [−1, 1] via cross-sectional ranking within each month
5. **Temporal splits** — strict past→future splits; no shuffling to prevent lookahead bias

### Feature Categories

| Category | Examples | Count |
|---|---|---|
| Momentum | mom1m, mom6m, mom12m, maxret | 7 |
| Liquidity | mvel1, dolvol, turn, ill, baspread | 8 |
| Risk/Volatility | retvol, idiovol, beta, betasq | 4 |
| Valuation | bm, ep, sp, cfp | 6 |
| Profitability | operprof, gma, roaq, roeq | 5 |
| Investment | agr, invest, grcapx, lgr | 4 |
| Other | age, pricedelay, acc | 2 |

### Target Variable
Forward 1-month return, constructed by shifting `mom1m` by −1 period within each stock (strictly per-stock to avoid cross-contamination).

### Train / Validation / Test Splits

```
Main experiment:
  Train:      2010 – 2017  (8 years)
  Validation: 2018 – 2019  (2 years)
  Test:       2020 – 2021  (2 years, COVID period)

Robustness check (Pre-COVID):
  Train:      2011 – 2015  (5 years)
  Validation: 2016 – 2017  (2 years)
  Test:       2018         (1 year, stable period)
```

### Hyperparameter Tuning Note
Elastic Net and Random Forest use `TimeSeriesSplit(n_splits=5)` inside `GridSearchCV` to ensure hyperparameter selection is free of lookahead bias — the validation fold is always strictly after the training fold.

---

## Project Structure

```
├── machine_learning_final_project.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── data/                                  # Not included — see Data section
    ├── datashare.csv                      # Main CRSP/Compustat panel (~4GB)
    └── permno_data.csv                    # PERMNO-to-company name mapping
```

---

## Data

The dataset is **not included** in this repository due to size (~4GB) and WRDS licensing.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
pathlib
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Limitations & Differences from Original Paper

| Aspect | Gu et al. (2020) | This Replication |
|---|---|---|
| Characteristics | 94 | 36 |
| Time period | 1957–2016 | 2010–2021 |
| Neural network depth | Up to NN5 (5 layers) | NN1 (3 layers) only |
| Portfolio construction | Full Sharpe ratio analysis | Not implemented |
| Interaction terms | Included | Not included |
| Hardware | High-performance cluster | Standard laptop |

---

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Campbell, J. Y., & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? *Review of Financial Studies*, 21(4), 1509–1531.
