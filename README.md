# FRED Mortgage Rate Forecasting
## Author
Patrick Lynch · DSE 6311 Capstone 

## Quick start
```bash
git clone https://github.com/lynchpo/dragon_exploder_fred.git
cd dragon_exploder_fred
python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt
jupyter notebook phase1_locked_walkthrough.ipynb
```
**Individual Capstone Project** – Independent Study  

**Goal:** Build an end-to-end forecasting pipeline for the 30-year fixed mortgage rate using Federal Reserve Economic Data (FRED), gradient boosting, (Phase 1) and Monte Carlo simulation for risk analysis (Phase 2).

## Project Structure
- `data/raw/` — Raw FRED data
- `data/processed/` — Cleaned and feature-engineered data
- `notebooks/` — EDA, modeling, and analysis notebooks
- `reports/` — Proposal, figures, and final report
- `src/` — Reusable functions and modules

## Key Components
- Data acquisition via FRED website
- Mixed-frequency time series handling
- Feature engineering (lags, rolling statistics, yield curve spreads)
- Gradient boosting models with proper time-series validation
- Monte Carlo simulation for risk metrics (e.g. maximum drawdown)

## Locked model (`src/phase1_locked_model.py`)

```python
from phase1_locked_model import (
    get_locked_params, load_feature_list, make_locked_lgbm, assert_exogenous,
)

model = make_locked_lgbm()
feats = load_feature_list()
assert_exogenous(feats)
```
## Self-test
- (2026-08-30, Colab): git clone into /content/dragon_exploder_fred;
- src/phase1_locked_model.py present; import succeeds; walkthrough runs
- FRED download → features → locked C.
- For a novel user: clone this repo, pip install -r requirements.txt, open from the repo root.

## Timeline
- Preliminary Proposal: Week of July 14
- Final Deliverables, Phase 1: August 27, 2026
