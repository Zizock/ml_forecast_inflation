# ==========================================================
# Evaluation module
# ==========================================================
# This script defines metrics, import test predictions,
# and runs the evaluation process across different models,
# then save the evaluation results to CSV files.
# ==========================================================

from __future__ import annotations
from dataclasses import dataclass

import pathlib
import numpy as np
import pandas as pd

from src.config import load_config, repo_path

# ==========================================================
# Define config dataclass for evaluation
# ==========================================================
@dataclass(frozen=True)
class config_eval:
    target_var: str
    horizons: list[int]
    models: list[str]
    results_root: pathlib.Path

def build_config_eval(config_path : pathlib.Path) -> config_eval:
    my_config = load_config(config_path)

    target_var = str(my_config.get("target_var", "X1"))
    horizons = list(my_config.get("forecasting_horizon", [1, 3, 6]))

    models = list(my_config.get("models", ["lasso", "xgb", "lstm", "lstnet"]))

    results_root = repo_path("results")

    return config_eval(
        target_var=target_var,
        horizons=horizons,
        models=models,
        results_root=results_root
    )

# ==========================================================
# Define evaluation metrics
# ==========================================================
# ==== safe MAPE with a small eps ====
def safe_mape(y_true: np.ndarray,
              y_pred: np.ndarray,
              eps: float = 1e-12
) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

# ==== symmetric MAPE: [0, 200] ====
def sym_mape(y_true: np.ndarray,
             y_pred: np.ndarray,
             eps: float = 1e-12
):
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

# ==== main metrics computation function ====
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray
) -> dict[str, float]:
    # forecast error
    err = y_pred - y_true
    # MAE
    mae = float(np.mean(np.abs(err)))
    # RMSE
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # calculate R2
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    return {
        "n": float(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": safe_mape(y_true, y_pred),
        "sym_mape_pct": sym_mape(y_true, y_pred),
        "bias": float(np.mean(err)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan, # if only one point, no corr
        "r2_oos": r2,
    }

# ==========================================================
# Read forecast results
# ==========================================================
# ==== load forecast results from a model ====
def read_test_result(results_root: pathlib.Path,
                     model_name: str
):
    """
    Loads: results/<model_name>/<model_name>_predictions_all_horizons.csv
    """
    path = results_root / model_name / f"{model_name}_predictions_all_horizons.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing file for model '{model_name}': {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df

# ==== get series for a specific horizon ====
def get_series(df: pd.DataFrame,
               target_var: str,
               h: int
):
    pred_col = f"{target_var}-pred-h{h}"
    true_col = f"{target_var}-true-h{h}"

    # a small check for missing columns
    missing = [c for c in (pred_col, true_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing columns for horizon {h}: {missing}"
        )

    aligned = pd.concat([df[true_col], df[pred_col]], axis=1).dropna()
    aligned.columns = ["y_true", "y_pred"]
    return aligned["y_true"].to_numpy(dtype=float), aligned["y_pred"].to_numpy(dtype=float)

# ==== pivot long evaluation results to wide format ====
def pivot_long_to_wide(eval_long: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in eval_long.columns if c not in ("model", "horizon")]
    wide = eval_long.pivot(index="model", columns="horizon", values=metrics)
    wide.columns = [f"{m}_h{h}" for (m, h) in wide.columns]
    return wide.reset_index()

# ==========================================================
# Main evaluation function
# ==========================================================
def run(config_path: pathlib.Path) -> dict:
    """
    Store results in:
      results/evaluation/evaluation_by_model_and_horizon.csv
      results/evaluation/evaluation_comparison_wide.csv

    Returns:
      dict with output paths
    """
    config_path = pathlib.Path(config_path)
    cfg = build_config_eval(config_path)

    out_dir = cfg.results_root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing_models: list[str] = []

    for model in cfg.models:
        try:
            df = read_test_result(cfg.results_root, model)
        # if file not found, skip
        except FileNotFoundError:
            missing_models.append(model)
            continue

        for h in cfg.horizons:
            y_true, y_pred = get_series(df, cfg.target_var, int(h))
            met = compute_metrics(y_true, y_pred)
            rows.append({"model": model, "horizon": int(h), **met})

    # another check: 
    if not rows:
        raise RuntimeError(
            "no evaluation rows produced. "
            f"missing models: {missing_models}. "
            f"expected files under: {cfg.results_root}"
        )

    # preserve model order in the config
    eval_long = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    eval_long["model"] = pd.Categorical(eval_long["model"], categories=cfg.models, ordered=True)
    eval_long = eval_long.sort_values(["model", "horizon"]).reset_index(drop=True)

    eval_wide = pivot_long_to_wide(eval_long)

    long_path = out_dir / "evaluation_by_model_and_horizon.csv"
    wide_path = out_dir / "evaluation_comparison_wide.csv"

    eval_long.to_csv(long_path, index=False)
    eval_wide.to_csv(wide_path, index=False)