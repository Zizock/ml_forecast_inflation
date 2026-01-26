# ==========================================================
# LSTM model for time series forecasting
# ==========================================================
# This script does the following:
# for a given back testing length (TEST_SIZE = 6),
# for each forecasting horizon in FORECASTING_HORIZON = [1,3,6]:
# tune and train an LSTM model with data from the start to (end - TEST_SIZE - h),
# then backtest on the last TEST_SIZE points
# save predictions and true values to results folder
# ==========================================================

import pathlib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.config import load_config, repo_path

# ==========================================================
# Output path and some configs
# ==========================================================
# The following constants are universal across different models:
# MODEL_NAME: LSTM
# RESULT_ROOT: results shared folder path / LSTM
# MANUAL_CUT_OFF_DATE: cut data for the entire train/test process
# MAX_LAG_FOR_FEATURES: max lag/diff operations for features (months), [not for RNN-based models]
# FORECASTING_HORIZON: list of forecasting horizons (months)
# MAX_HORIZON: max forecasting horizon (months)
# TEST_SIZE: test size (months)
# TARGET_VAR: target variable name (X1 is my inflation series)
# RANDOM_SEED: random seed for reproducibility

# SEQUENCE_LENGTH: input sequence length (months), for RNN-based models
# optuna tuning configs:
# N_TRIALS: N different trials
# MAX_EPOCHS_TUNE: max epochs for each trial during tuning
# MAX_EPOCHS_REFIT: max epochs for refitting the best model

MODEL_NAME = "LSTM"
RESULT_ROOT = repo_path("results", MODEL_NAME)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

MANUAL_CUT_OFF_DATE = pd.to_datetime(my_config["manual_cut_off_date"])

MAX_LAG_FOR_FEATURES = my_config["max_lag_for_features"] # months, not used in LSTM but kept for consistency with XGBoost
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
# Data cutting into train/val/test sets
# ==========================================================
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
df = df.dropna().copy() # drop NA rows
df = df.loc[df.index < MANUAL_CUT_OFF_DATE]
# split into train/val/test based on date
# this split is the same for all horizons
train_val_data = df.iloc[:-TEST_SIZE]
train_data, val_data = train_val_data.iloc[:-TEST_SIZE], train_val_data.iloc[-TEST_SIZE:]
test_data = df.iloc[-TEST_SIZE:]

# ==========================================================
# Build sliding fixed window dataset class for horizon h
# ==========================================================
class SlidingFixedWindow(Dataset):
    """
    Args:
        data: a df
        seq_length: length of per sequence in training
        target_var: target variable column name (X1 for inflation)
        horizon: forecasting horizon h (months)
    """
    def __init__(self, data, seq_length, target_var, horizon = 1):
        self.data = data
        self.seq_length = seq_length
        self.target_var = target_var
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_length - self.horizon + 1
        # number of sequence starting points that can be extracted
    
    def __getitem__(self, index):
        return(
            torch.tensor( # input sequence X
                self.data.iloc[index : index + self.seq_length].values,
                dtype = torch.float),
            torch.tensor( # output y
                [self.data.iloc[index + self.seq_length + self.horizon - 1][self.target_var]],
                dtype=torch.float)
        )

# and a make_loaders function for each horizon h, and each batch_size during tuning
def make_loaders(train_df, train_val_df, h, batch_size):
    """
    place holder. add docstring later if I feel needed
    """
    train_dataset = SlidingFixedWindow(train_df,
                                       seq_length = SEQUENCE_LENGTH,
                                       target_var = TARGET_VAR,
                                       horizon = h)
    
    val_dataset = SlidingFixedWindow(train_val_df[-(SEQUENCE_LENGTH + TEST_SIZE):],
                                seq_length = SEQUENCE_LENGTH,
                                target_var=TARGET_VAR,
                                horizon=h)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader

# ==========================================================
# Build LSTM model using PyTorch Lightning
# ==========================================================
class LSTM_model(L.LightningModule):
    """
    Args:
        input_dim (int): Number of time series df.shape[1]
        hidden_dim (int): LSTM hidden state dimension
        num_layers (int): number of LSTM layers
        dropout (float): dropout rate between LSTM layers (when num_layers > 1)
        output_dim (int): output dimension (1 in this project because its only inflation)
        learning_rate (float): learning rate for Adam optimizer
        weight_decay (float): weight decay for AdamW optimizer
        loss_name (str): "mae" or "huber" loss function (I use mae here, maybe try huber later)
    """
    def __init__(
            self,
            input_dim,
            hidden_dim = 64,
            num_layers = 1,
            dropout = 0.0,
            output_dim = 1,
            learning_rate = 1e-3,
            weight_decay = 0.0,
            loss_name = "mae",
            ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=(dropout if num_layers > 1 else 0.0),
                                  )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # define loss function (MAE or Huber)
        # temporarily only use MAE in this project but leave here for future extension
        if loss_name == "huber":
            self.loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
        else:
            self.loss_fn = torch.nn.L1Loss()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X) # self calls forward() function and get estimated outputs
        loss = self.loss_fn(yhat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X)
        loss = self.loss_fn(yhat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    # added AdamW and try a w_d=0, which is equivalent to Adam
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)

# ==========================================================
# Define training and tuning with optuna
# ==========================================================
def tune_with_optuna(train_df, train_val_df, h):
    """
    Args:
        train_df: training data DataFrame
        train_val_df: training + validation data DataFrame
        h: forecasting horizon
    
    Returns:
        study.best_params (a dict)
    """
    def objective(trial):
        # search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-2, log=True)
        hidden_dim    = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
        num_layers    = trial.suggest_int("num_layers", 1, 3)
        dropout       = trial.suggest_float("dropout", 0.0, 0.5)

        # try both Adam(wd=0) and AdamW
        use_weight_decay = trial.suggest_categorical(
            "use_weight_decay", [False, True]
        )
        if use_weight_decay:
            weight_decay = trial.suggest_float(
                "weight_decay", 1e-8, 1e-3, log=True
            )
        else:
            weight_decay = 0.0

        batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 20])
        grad_clip  = trial.suggest_float("grad_clip", 0.5, 5.0)
        #loss_name     = trial.suggest_categorical("loss", ["mae", "huber"])

        # load data using the suggested batch_size
        train_loader, val_loader = make_loaders(train_df, train_val_df, h, batch_size)

        model = LSTM_model(
            input_dim=train_val_df.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            #loss_name=loss_name,
        )

        callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ]

        trainer = L.Trainer(
        max_epochs=MAX_EPOCHS_TUNE,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
        deterministic=True,
        gradient_clip_val=grad_clip,
        log_every_n_steps=10,
        )

        trainer.fit(model, train_loader, val_loader)

        val = trainer.callback_metrics.get("val_loss")

        # a quick check to avoid returning None or nan to optuna
        if val is None:
            return float("inf")
        val = val.item()
        if not np.isfinite(val):
            return float("inf")
        return val
    
    # define and start the study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))
    return study.best_params

# ==========================================================
# Refit the model on entire train/val dataset with best hyperparameters
# ==========================================================
# output below is a tuple of (best_model, best_path)
def refit_best_model(train_val_df, h, best_hps):

    seed_everything(RANDOM_SEED, workers=True)
    
    # the final dataset for training the best model
    final_df = SlidingFixedWindow(train_val_df, SEQUENCE_LENGTH, TARGET_VAR, horizon=h)
    final_loader = DataLoader(final_df, batch_size=best_hps["batch_size"], shuffle=True)

    model = LSTM_model(
        input_dim=train_val_df.shape[1],
        output_dim=1,
        hidden_dim=best_hps["hidden_dim"],
        num_layers=best_hps["num_layers"],
        dropout=best_hps["dropout"],
        learning_rate=best_hps["learning_rate"],
        weight_decay = best_hps.get("weight_decay", 0.0) if best_hps.get("use_weight_decay", False) else 0.0,
        loss_name=best_hps["loss"],   # if added loss_name it makes sense but not yet
    )

    # save the last checkpoint (no val_loader here because it is a refit)
    ckpt = ModelCheckpoint(
        save_top_k=1,
        monitor=None, # nothing ot monitor because no val_loader
        save_last=True, # keep last.ckpt
        filename=f"last-h{h}" + "-{epoch:03d}",
    )

    final_trainer = L.Trainer(
        max_epochs=MAX_EPOCHS_REFIT,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ckpt # no early stopping during refit
        ],
        deterministic=True,
        gradient_clip_val=best_hps.get("grad_clip", 1.0), # default to 1 if not found
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=10,
    )
    final_trainer.fit(model, final_loader)
    best_path = ckpt.last_model_path
    best_model = LSTM_model.load_from_checkpoint(best_path, **model.hparams)
    # **model.hparams not a necessary step but just to be sure I have the right hyperparameters
    best_model.eval()
    return best_model, best_path

# ==========================================================
# Forecast on test set
# ==========================================================

def forecast_on_test(df_h, test_df, model, h):
    """
    Args:
        df_h: the entire dataset (DF)
        test_df: test data (DF)
        model: best LSTM_model from refitting
        h: forecasting horizon
    """
    model.eval() # set to eval mode
    device = next(model.parameters()).device # get model device (cpu or gpu)

    preds = [] # initialte a result container

    with torch.no_grad():
        test_size = test_df.shape[0]
        for idx in range(test_size):
            X = torch.tensor(
                df_h.iloc[-(SEQUENCE_LENGTH + test_size) -h+1 + idx : - test_size -h+1 + idx].values,
                dtype=torch.float
                ).unsqueeze(0).to(device) # unsqueeze to add batch dimension and move to device
            yhat = model(X)
            preds.append(yhat.item()) # get scalar value and append to list
    
    preds = torch.tensor(preds) # convert a list of multiple tensors to a single tensor

    final_out = pd.DataFrame(
        {
            f"{TARGET_VAR}-pred-h{h}": np.asarray(preds), # predicted values converted to numpy array
            f"{TARGET_VAR}-true-h{h}": np.asarray(test_df[TARGET_VAR])
        },
        index=test_df.index
    )
    final_out.index.name = "Date"
    return final_out

# ==========================================================
# Main function to run all horizons
# ==========================================================
def main():
    seed_everything(RANDOM_SEED, workers=True)

    all_out = []
    for h in FORECASTING_HORIZON:
        best_hps = tune_with_optuna(train_df = train_data,
                                    train_val_df = train_val_data, 
                                    h = h)
        
        model, _ = refit_best_model(train_val_df = train_val_data, # the _ is for best_path which is not used here (yet)
                                    h = h, 
                                    best_hps = best_hps)

        out_h = forecast_on_test(df_h = df, test_df = test_data, model = model, h = h)
        out_h.to_csv(RESULT_ROOT / f"{MODEL_NAME}_predictions_h{h}.csv", index=True)
        all_out.append(out_h)
    
    combined = pd.concat(all_out, axis=1)
    combined.to_csv(RESULT_ROOT / f"{MODEL_NAME}_predictions_all_horizons.csv", index=True)

# make it import-safe, but now is not import-safe yet
if __name__ == "__main__":
    main()