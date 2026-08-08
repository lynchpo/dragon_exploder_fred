"""
Phase 1 locked model — Dragon Exploder / FRED mortgage forecast
Model ID: C_fixed500
Freeze date: 2026-08-08

Usage:
    from phase1_locked_model import get_locked_params, load_feature_list, make_locked_lgbm
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

LOCKED_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 31,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}

TARGET = "MORTGAGE30US"
MODEL_ID = "C_fixed500"
TRAIN_END = "2021-03-05"
TEST_START = "2021-03-12"


def get_locked_params() -> Dict[str, Any]:
    """Return a copy of Phase 1 locked LightGBM hyperparameters."""
    return dict(LOCKED_PARAMS)


def load_feature_list(path: str | Path | None = None) -> List[str]:
    """Load exogenous feature names (excludes target; no MORTGAGE30US_*)."""
    if path is None:
        path = Path(__file__).with_name("phase1_feature_list.txt")
    path = Path(path)
    feats = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        feats.append(line)
    assert TARGET not in feats
    assert not any(f.startswith("MORTGAGE30US_") for f in feats)
    return feats


def make_locked_lgbm():
    """Instantiate LightGBM regressor with locked Phase 1 hyperparameters."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(**get_locked_params())


def assert_exogenous(feature_names: List[str]) -> None:
    """Raise if target-derived columns appear."""
    bad = [f for f in feature_names if f == TARGET or f.startswith("MORTGAGE30US_")]
    if bad:
        raise ValueError(f"Target leakage columns present: {bad}")
