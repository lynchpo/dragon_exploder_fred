# dragon_exploder_fred
Individual Capstone Project: Forecasting 30-Year Mortgage Rates using FRED data and Gradient Boosting
# FRED Mortgage Rate Forecasting

**Individual Capstone Project** – Independent Study  
**Goal:** Build an end-to-end forecasting pipeline for the 30-year fixed mortgage rate using Federal Reserve Economic Data (FRED), gradient boosting, and Monte Carlo simulation for risk analysis.

## Project Structure
- `data/raw/` — Raw FRED data
- `data/processed/` — Cleaned and feature-engineered data
- `notebooks/` — EDA, modeling, and analysis notebooks
- `reports/` — Proposal, figures, and final report
- `src/` — Reusable functions and modules

## Key Components
- Data acquisition via `fredapi`
- Mixed-frequency time series handling
- Feature engineering (lags, rolling statistics, yield curve spreads)
- Gradient boosting models with proper time-series validation
- Monte Carlo simulation for risk metrics (e.g. maximum drawdown)

## Timeline
- Preliminary Proposal: Week of July 14
- Final Deliverables: August 27, 2026

## Author
Patrick
