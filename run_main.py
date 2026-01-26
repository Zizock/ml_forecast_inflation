# ==========================================================
# Main script to run the entire workflow: data processing, training, and evaluation
# ==========================================================
# TO BE COMPLETED DO NOT RUN
# currently models are not import-safe

from pathlib import Path

from models.model_Lasso import run_it as run_lasso
from models.model_XGB import run_it as run_xgb
from models.model_LSTM import run_it as run_lstm
from models.model_LSTNet import run_it as run_lstnet

def main():

    run_lasso()
    run_xgb()
    run_lstm()
    run_lstnet()

if __name__ == "__main__":
    main()