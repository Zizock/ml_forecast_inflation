# ==========================================================
# Graphing module
# ==========================================================
# This script draws RMSE comparison graphs from evaluation results
# ==========================================================

from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config, repo_path

# ==========================================================
# Define config dataclass for graphing
# ==========================================================
@dataclass(frozen=True)
class config_graph:
    models: list[str]
    horizons: list[int]

def build_config_graph(config_path: Path) -> config_graph:
    my_config = load_config(config_path)

    models = list(my_config.get("models",
                                ["lasso", "xgb", "lstm", "lstnet"]))
    horizons = list(my_config.get("forecasting_horizon", [1, 3, 6]))

    return config_graph(
        models=models,
        horizons=horizons,
    )

# ==========================================================
# Define graphing functions
# ==========================================================
def plot_rmse_comparison(df: pd.DataFrame,
                         cfg: config_graph,
                         out_path: Path | None = None,
                         title: str = "Forecasting RMSE"
) -> None:
    """
    Plot grouped bar chart comparing RMSE across models and horizons.

    Args:
    df: pd.DataFrame that the evaluation module produced
    cfg: config_graph
    out_path: path to save the figure, None = do not save
    title: as its name suggests
    """

    df = df.copy()
    df["model"] = df["model"].astype(str).str.strip().str.lower() # just in case some have capital letters

    models = [m.lower() for m in cfg.models]
    horizons = cfg.horizons

    # Build rmse matrix: rows=models, cols=horizons
    rmse = pd.DataFrame(
        {
            f"h{h}": df.set_index("model")[f"rmse_h{h}"]
            for h in horizons
        }
    ).reindex(models) # match order of list models

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1) # just in case no models

    for i, model in enumerate(models):
        ax.bar(
            x + i * width,
            rmse.loc[model].values,
            width,
            label=model.upper(),
        )

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{h}M ahead" for h in horizons])
    ax.set_ylabel("RMSE")
    ax.set_title(title)

    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

    plt.close(fig)

# ==========================================================
# Main function
# ==========================================================
def run(config_path: Path,
        eval_retrieve_path: Path | None = None,
        out_path: Path | None = None,
) -> Path:
    """
    Draw RMSE comparison graph and save to file.
    
    Args:
    config_path: path of my config.yaml
    eval_retrieve_path: path to evaluation generated CSV
    out_path: output figure saving path; defaults to repo standard location

    Returns: path to saved figure
    """
    cfg = build_config_graph(Path(config_path))

    eval_retrieve_path = Path(eval_retrieve_path) if eval_retrieve_path is not None else repo_path(
        "results/evaluation/evaluation_comparison_wide.csv"
    )
    out_path = Path(out_path) if out_path is not None else repo_path(
        "results/figures/rmse_comparison.png"
    )

    df = pd.read_csv(eval_retrieve_path)

    plot_rmse_comparison(
        df=df,
        cfg=cfg,
        out_path=out_path,
        title="Forecasting RMSE",
    )

    return out_path


if __name__ == "__main__":
    run()