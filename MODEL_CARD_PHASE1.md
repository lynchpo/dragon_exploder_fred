# Model Card — Phase 1 Locked Model (`C_fixed500`)

**Project:** Economic Forecasting for Banking Using FRED Time Series (Dragon Exploder)  
**Freeze date:** 2026-08-08  
**Owner:** Patrick Lynch (`lynchpo/dragon_exploder_fred`)

## Decision task

Recover **exogenous** macro signal for a bank-facing planning/risk narrative under a leakage-controlled design.  
**Not** a claim that the model beats a random walk at pure one-week-ahead persistence.

## Model

| Item | Value |
|------|--------|
| Algorithm | LightGBM regressor |
| Model ID | `C_fixed500` |
| Target | `MORTGAGE30US` (weekly) |
| Features | See `phase1_feature_list.txt` (exogenous only) |
| learning_rate | 0.05 |
| max_depth | 4 |
| num_leaves | 31 |
| n_estimators | 500 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| random_state | 42 |

## Selection protocol

Three train-only candidates scored **once** on chronological hold-out (2021-03-12 → 2026-06-05):

| ID | Rule | Result |
|----|------|--------|
| A | Pre-tune reference | Test R² 0.833 |
| B | Early-stop on calm val | Test R² 0.789 |
| **C** | **Best train val MSE, fixed 500 trees** | **Test R² 0.845** |

**Best definition:** minimize `Test_MSE + Stress_MSE(2022–2023)`.

## Hold-out metrics (C)

| Metric | Value |
|--------|-------|
| Test R² | 0.845 |
| Test MSE | 0.320 |
| Test MAE | 0.477 |
| Mean residual | +0.400 |
| 2022–23 mean residual | +0.714 |
| Co-primary score | 0.906 |

## Feature rule

- No `MORTGAGE30US_*` lags, rolling stats, or other target-derived columns.
- Mixed-frequency alignment: weekly Friday; monthly series forward-filled after resample.
- Restricted ladder: DGS10_lag1 only → Treasury core → full exogenous (full wins overall score).

## Limits

- Systematic under-prediction in 2022–2023 tightening window remains.
- Does not beat random walk at one-step horizon.
- Predictive intervals under regime shift are not calibrated (see Bayesian Ridge coverage failure in baseline report).

## Files in this freeze

- `phase1_locked_config.json` — machine-readable config + metrics
- `phase1_feature_list.txt` — ordered exogenous feature names
- `phase1_locked_model.py` — `get_locked_params()`, `load_feature_list()`, `make_locked_lgbm()`
- `MODEL_CARD_PHASE1.md` — this card

## Reproduce (sketch)

```python
from phase1_locked_model import make_locked_lgbm, load_feature_list, TARGET, assert_exogenous
features = load_feature_list()
assert_exogenous(features)
model = make_locked_lgbm()
model.fit(X_train[features], y_train)
```
