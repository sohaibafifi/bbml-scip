from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional
    xgb = None
    _HAS_XGB = False


@dataclass
class BaselineConfig:
    features: List[str]
    learning_rate: float = 0.1
    max_depth: int = 6
    max_iter: int = 200


def prepare_pairs(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare (X, y) where y is a per-candidate target score.

    If sb_score/sb_score_up/down exist, uses them. Otherwise, prefers a pseudocost
    proxy when present, and falls back to |reduced_cost|.
    """
    X = df[features].astype(float).values
    if "sb_score" in df.columns:
        y = df["sb_score"].astype(float).values
    elif "sb_score_up" in df.columns:
        up = df["sb_score_up"].astype(float).values
        if "sb_score_down" in df.columns:
            down = df["sb_score_down"].astype(float).values
            y = np.maximum(up, down)
        else:
            y = up
    elif {"pseudocost_up", "pseudocost_down"}.issubset(df.columns):
        y = np.maximum(
            df["pseudocost_up"].astype(float).values,
            df["pseudocost_down"].astype(float).values,
        )
    else:
        y = df["reduced_cost"].abs().astype(float).values
    return X, y


def fit_hist_gbr(df: pd.DataFrame, cfg: BaselineConfig):
    X, y = prepare_pairs(df, cfg.features)
    model = HistGradientBoostingRegressor(max_depth=cfg.max_depth, max_iter=cfg.max_iter, learning_rate=cfg.learning_rate)
    model.fit(X, y)
    return model


def fit_xgboost(df: pd.DataFrame, cfg: BaselineConfig):
    """Fit XGBoost regressor if available; fall back to HGBR otherwise."""
    if not _HAS_XGB:
        return fit_hist_gbr(df, cfg)
    X, y = prepare_pairs(df, cfg.features)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": cfg.max_depth,
        "eta": cfg.learning_rate,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "verbosity": 0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=cfg.max_iter)
    return bst
