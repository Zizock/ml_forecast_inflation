# ==========================================================
# Main script to run the training and evaluation of models
# ==========================================================

from __future__ import annotations
from pathlib import Path

from src.models.module_lasso import run as run_lasso
from src.models.module_xgb import run as run_xgb
from src.models.module_lstm import run as run_lstm
from src.models.module_lstnet import run as run_lstnet
from src.evaluation import run as run_eval

# ==== define paths ====
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config" / "my_config.yaml"

# ==== define models to run ====
MODEL_DICT = {
    "lasso": run_lasso,
    "xgb": run_xgb,
    "lstm": run_lstm,
    "lstnet": run_lstnet,
    "evaluation": run_eval, # evaluation cycle
}

# ==== main function ====
def main(config_path=CONFIG_FILE):
    results = []

    for model_name, run_func in MODEL_DICT.items():
        print(f"\n==== Running model: {model_name} ====")
        result = run_func(config_path)
        results.append(result)
        print(f"==== Finished model: {model_name} ====\n")

    print("\n==== All models done ====")
    for r in results:
        if "combined_csv" in r:
            print(r["model"], "->", r["combined_csv"]) # name of the combined csv file
        else:
            print(r["model"], "->", r["result_dir"]) # path to result dir

# ==== entry point ====
if __name__ == "__main__":
    main()