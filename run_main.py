# ==========================================================
# Main script to run the training and evaluation of models
# ==========================================================

from __future__ import annotations
from pathlib import Path

# models
from src.models.module_lasso import run as run_lasso
from src.models.module_xgb import run as run_xgb
from src.models.module_lstm import run as run_lstm
from src.models.module_lstnet import run as run_lstnet

# evaluation module
from src.evaluation import run as run_eval

# graphing module
from src.graphrmse import run as run_graph

# ==== define paths ====
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config" / "my_config.yaml"

# ==== define models to run ====
MODEL_DICT = {
    "lasso": run_lasso,
    "xgb": run_xgb,
    "lstm": run_lstm,
    "lstnet": run_lstnet,
}

# ==== main function ====
def main(config_path=CONFIG_FILE):
    results = []

    # run models
    for model_name, run_func in MODEL_DICT.items():
        print( f"\n==== Running model: {model_name} ====" )
        result = run_func(config_path)
        results.append(result)
        print( f"==== Finished model: {model_name} ====\n" )
    print(results)
    print( "\n==== All models done ====" )

    # run evaluation
    print( "\n==== Running evaluation ====" )
    run_eval(config_path)
    print( "==== Finished evaluation ====\n" )

    # run graphing
    print( "\n==== Drawing graph ====" )
    run_graph(config_path)
    print( "==== Saved graph ====\n" )

# ==== entry point ====
if __name__ == "__main__":
    main()