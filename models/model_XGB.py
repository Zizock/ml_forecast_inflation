# ==========================================================
# XGBoost model for time series forecasting
# ==========================================================
# This script does the following:
# for a given back testing length (TEST_SIZE = 6),
# for each forecasting horizon in FORECASTING_HORIZON = [1,3,6]:
# does times series feature engineering for the supervised method,
# tune and train an XGBoost model with data from the start to (end - TEST_SIZE - h),
# then backtest on the last TEST_SIZE points
# save predictions and true values to results folder
# ==========================================================

import pathlib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import optuna

from src.config import load_config, repo_path

# ==========================================================
# Output path and some configs
# ==========================================================
# The following constants are universal across different models:
# MODEL_NAME: XGBoost
# RESULT_ROOT: results shared folder path / XGBoost
# MANUAL_CUT_OFF_DATE: cut data for the entire train/test process
# MAX_LAG_FOR_FEATURES: max lag/diff operations for features (months)
# FORECASTING_HORIZON: list of forecasting horizons (months)
# TEST_SIZE: test size (months)
# TARGET_VAR: target variable name (X1 is my inflation series)
# RANDOM_SEED: random seed for reproducibility

MODEL_NAME = "XGBoost"
RESULT_ROOT = repo_path("results", MODEL_NAME)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

MANUAL_CUT_OFF_DATE = pd.to_datetime(my_config["manual_cut_off_date"])

MAX_LAG_FOR_FEATURES = my_config["max_lag_for_features"] # months
FORECASTING_HORIZON = my_config["forecasting_horizon"] # a list of months
MAX_HORIZON = max(FORECASTING_HORIZON)

TEST_SIZE = my_config["test_size"] # months
TARGET_VAR = my_config["target_var"]
RANDOM_SEED = my_config["random_seed"]

# RNN specific configs
SEQUENCE_LENGTH = my_config["sequence_length_factor"] * MAX_HORIZON # input sequence length (months)

# optuna tuning configs
N_TRIALS = my_config["n_trials"]
MAX_EPOCHS_TUNE = my_config["max_epochs_tune"]
MAX_EPOCHS_REFIT = my_config["max_epochs_refit"]

# Dataloader
NUM_WORKERS = my_config["num_workers"]

DATA_FILE = repo_path("data", "processed_data.csv")

# ==========================================================
# Feature engineering (excl. Y_t generation which depends on horizon)
# ==========================================================
# ==== add lag and rolling mean/std features ====
def add_lag_and_rolling_features(data, max_lag) -> pd.DataFrame:
    """
    Args:
        data: original data df
        max_lag: maximum lag to create -> MAX_LAG_FOR_FEATURES
        
    Add lag and rolling mean/std features for each column.
    """
    lag_data = {}
    for col in data.columns:
        for lag in range(1, max_lag + 1):
            lag_data[f"{col}-lag-{lag}"] = data[col].shift(lag)

        window = max_lag + 1
        lag_data[f"{col}-rollingmean"] = data[col].rolling(window).mean()
        lag_data[f"{col}-rollingstd"] = data[col].rolling(window).std()

    lag_df = pd.DataFrame(lag_data, index=data.index)
    return pd.concat([data, lag_df], axis=1)

# ==== add date features ====
def add_date_features(index) -> pd.DataFrame:
    """
    Args:
        index: datetime index of the data
        
    Create quarter/month features from datetime index.
    """
    quarters = pd.DataFrame(index.quarter.values, index=index, columns=["quarter"])
    months = pd.DataFrame(index.month.values, index=index, columns=["month"])
    return pd.concat([quarters, months], axis=1)

# ==========================================================
# Build dataset for a specific forecasting horizon
# ==========================================================
def build_horizon_dataset(
    data_with_features, # the engineered lag/rolling features above
    date_features, # the date features above
    target_var, # TARGET_VAR defined at the beginning
    h, # forecasting h months ahead
    ) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """
    Build df_h = X features + date features + Y target (shifted by -h).
    Returns:
        df_h, X_columns, num_X_columns, Y_column (lists of column names)
    """
    y = data_with_features[[target_var]].shift(-h)
    y_col = f"{target_var}-Y_t_h{h}"
    y.columns = [y_col]

    df_h = pd.concat([data_with_features, date_features, y], axis=1)

    date_columns = ["quarter", "month"]
    Y_column = [y_col]

    # numeric X columns = everything except date and target
    # X columns = numeric X + date
    num_X_columns = [col for col in df_h.columns if col not in date_columns + Y_column]
    X_columns = num_X_columns + date_columns

    df_h = df_h.dropna().copy()
    # added here: cut df to the designed length (after all feature engineering done)
    df_h = df_h.loc[df_h.index < MANUAL_CUT_OFF_DATE]
    return df_h, X_columns, num_X_columns, date_columns, Y_column

# ==========================================================
# Split data into train/val/test according to the horizon h
# ==========================================================
def split_train_test(df_h,
                     h, # forecasting h months ahead
                     test_size) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by last test_size rows as test."""
    # a quick check on test_size validity
    if test_size <= 0 or test_size >= len(df_h):
        raise ValueError(f"Invalid TEST_SIZE={test_size} for df length={len(df_h)}")
    return df_h.iloc[:-test_size-h].copy(), df_h.iloc[-test_size-h:-h].copy()

# ==========================================================
# Define preprocessing and model pipeline
# ==========================================================
def make_pipeline(num_X_columns, # a list of numeric feature column names
                  date_columns, # a list of date feature column names
                  seed) -> Pipeline: # let it be 42
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_X_columns),
            ("cat", OrdinalEncoder(), date_columns),
        ]
    )

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        enable_categorical=True,
        eval_metric="mae",
        n_jobs=1, # I use -1 in cross_val_score to use all cores, here set to 1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb_model),
        ]
    )
    return pipeline

# ==========================================================
# Define training and tuning with optuna
# ==========================================================

def tune_hyperparams_optuna(
    pipeline, # a pipeline defined before
    X, # input df
    y, # target series
    n_trials = 50,
    n_splits = 5,
    seed = 42,
    ) -> tuple[dict, float]:
    """
    Tune hps with optuna using TimeSeriesSplit CV on MAE.
    Returns (best_params_dict_for_pipeline, best_mae)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial) -> float:
        params = {
            # bossting parameters
            "regressor__n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
            "regressor__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),

            # tree parameters
            "regressor__max_depth": trial.suggest_int("max_depth", 2, 8),
            "regressor__min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "regressor__gamma": trial.suggest_float("gamma", 0.0, 10.0),

            # subsampling parameters
            "regressor__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "regressor__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "regressor__colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),

            # regularization parameters
            "regressor__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "regressor__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }

        model = pipeline.set_params(**params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        return -scores.mean() # convert to positive MAE

    # define and start the study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_mae = study.best_value
    best_params_for_pipeline = {f"regressor__{k}": v for k, v in study.best_params.items()}
    return best_params_for_pipeline, best_mae

# ==========================================================
# Back testing
# ==========================================================
def expanding_window_backtest(
    best_model, # a fitted pipeline
    train_val_data,
    test_data,
    X_columns,
    Y_column,
    ):
    """
    Expanding window backtest:
        1, for each test point, refit on (train_val + earlier test points), then predict current point.
        2, returns a Series aligned with test_data.index.
    """
    preds = []
    for idx in range(test_data.shape[0]):
        data_until_t = pd.concat([train_val_data, test_data.iloc[:idx]], axis=0)

        X_input = data_until_t[X_columns]
        y_input = data_until_t[Y_column].squeeze() # make it 1d array

        model = clone(best_model)
        model.fit(X_input, y_input)

        X_test = test_data[X_columns].iloc[idx:idx+1]
        y_pred = model.predict(X_test)[0]
        preds.append(y_pred)

    return pd.Series(preds, index=test_data.index)

# ==========================================================
# Run for a specific horizon
# ==========================================================
def run_one_horizon(
    data_features, 
    date_features,
    target_var,
    h,
    test_size,
    n_trials,
    seed,
    ) -> pd.DataFrame:

    # build dataset for horizon h and collects column names
    df_h, X_columns, num_X_columns, date_columns, Y_column = build_horizon_dataset(
        data_with_features=data_features,
        date_features=date_features,
        target_var=target_var,
        h=h,
    )

    # split train/val and test sets
    train_val_data, test_data = split_train_test(df_h, h=h, test_size=test_size)

    # feed into pipeline
    pipeline = make_pipeline(num_X_columns=num_X_columns, date_columns=date_columns, seed=seed)

    # form X and y for training
    X = train_val_data[X_columns]
    y = train_val_data[Y_column].squeeze() # make it a 1d array

    # hyperparameter tuning
    best_params, best_mae = tune_hyperparams_optuna(
        pipeline=pipeline,
        X=X,
        y=y,
        n_trials=n_trials,
        n_splits=5,
        seed=seed,
    )

    # refit best model on entire train_val set
    best_model = pipeline.set_params(**best_params)
    best_model.fit(X, y)

    # backtest on test set
    preds = expanding_window_backtest(
        best_model=best_model,
        train_val_data=train_val_data,
        test_data=test_data,
        X_columns=X_columns,
        Y_column=Y_column,
    )

    # output: a DataFrame with predictions and true values
    pred_index = df_h.index[-len(test_data) : ] # length is also test_size
    out = pd.DataFrame(
        {
            f"{target_var}-pred_h{h}": np.asarray(preds), # make sure it is np array
            f"{target_var}-true_h{h}": test_data[Y_column].squeeze().values, # make it 1d array
        },
        index=pred_index, # forcely use correct index matching the original data
    )

    print(f"h={h}: best CV MAE={best_mae:.6f}")
    return out

# ==========================================================
# Main function to run all horizons
# ==========================================================
def main():
    # load data
    data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

    # feature engineering once
    data_features = add_lag_and_rolling_features(data, max_lag=MAX_LAG_FOR_FEATURES)
    date_features = add_date_features(data_features.index)

    all_out = []
    for h in FORECASTING_HORIZON: # loop over all forecasting horizons
        out_h = run_one_horizon(
            data_features=data_features,
            date_features=date_features,
            target_var=TARGET_VAR,
            h=h,
            test_size=TEST_SIZE,
            n_trials=N_TRIALS,
            seed=RANDOM_SEED,
        )
        out_h.to_csv(RESULT_ROOT / f"{MODEL_NAME}_predictions_h{h}.csv", index=True)
        all_out.append(out_h)

    combined = pd.concat(all_out, axis=1)
    combined.to_csv(RESULT_ROOT / f"{MODEL_NAME}_predictions_all_horizons.csv", index=True)

# make it import-safe, but now is not import-safe yet
if __name__ == "__main__":
    main()