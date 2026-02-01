# ==========================================================
# A callable LSTM module for time series forecasting
# ==========================================================
# This script does the following:
# for a given back testing length,
# for each forecasting horizon,
# tune and train an LSTM model with data from train/val period,
# then backtest on the last few points
# save predictions and true values to results folder
# ==========================================================

from __future__ import annotations
from dataclasses import dataclass

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
# Define config dataclass and build from yaml
# ==========================================================
# ==== define config dataclass ====
@dataclass(frozen=True)
class config_lstm:
    model_name: str
    result_root: pathlib.Path
    data_file: pathlib.Path

    manual_cut_off_date: pd.Timestamp

    forecasting_horizon: list[int]

    test_size: int
    target_var: str
    random_seed: int

    sequence_length_factor: int

    n_trials: int
    max_epochs_tune: int
    max_epochs_refit: int
    num_workers: int

# ==== build config dataclass for LSTM ====
def build_config_lstm(config_path : pathlib.Path) -> config_lstm:
    my_config = load_config(config_path)
    model_name = "lstm"
    result_root = repo_path("results", model_name)
    result_root.mkdir(parents=True, exist_ok=True)

    return config_lstm(
        model_name=model_name,
        result_root=result_root,
        data_file=repo_path("data", "processed_data.csv"),
        manual_cut_off_date=pd.to_datetime(my_config["manual_cut_off_date"]),
        forecasting_horizon=list(my_config["forecasting_horizon"]),
        test_size=int(my_config["test_size"]),
        target_var=str(my_config["target_var"]),
        random_seed=int(my_config["random_seed"]),
        sequence_length_factor=int(my_config["sequence_length_factor"]),
        n_trials=int(my_config["n_trials"]),
        max_epochs_tune=int(my_config["max_epochs_tune"]),
        max_epochs_refit=int(my_config["max_epochs_refit"]),
        num_workers=int(my_config["num_workers"]),
    )

# ==========================================================
# Data cutting into train/val/test sets
# ==========================================================
def split_train_test(df : pd.DataFrame,
                     manual_cut_off_date : pd.Timestamp,
                     test_size : int):
    """
    Split data into train/val and test sets based on manual cut-off date and test size

    Args:
        df (pd.DataFrame): input dataframe with Date index
        manual_cut_off_date (pd.Timestamp): the raw df needs a prune
        test_size (int): number of periods for test set

    Returns:
        pd.DataFrame, pd.DataFrame: train/val and test sets
    """
    df = df.dropna().copy()
    df = df.loc[df.index < manual_cut_off_date]

    train_val_data = df.iloc[:-test_size]
    train_data, val_data = train_val_data.iloc[:-test_size], train_val_data.iloc[-test_size:]
    test_data = df.iloc[-test_size:]

    return df, train_val_data, train_data, val_data, test_data

# ==========================================================
# Build sliding fixed window dataset class for horizon h
# ==========================================================
class SlidingFixedWindow(Dataset):
    """
    Args:
        data: a df
        sequence_length: length of per sequence in training
        target_var: target variable column name (X1 for inflation)
        horizon: forecasting horizon h (months)
    """
    def __init__(self, data : pd.DataFrame,
                 sequence_length: int,
                 target_var: str,
                 horizon: int,):
        self.data = data
        self.sequence_length = sequence_length
        self.target_var = target_var
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.horizon + 1
        # number of sequence starting points that can be extracted
    
    def __getitem__(self, index):
        return(
            torch.tensor( # input sequence X
                self.data.iloc[index : index + self.sequence_length].values,
                dtype = torch.float),
            torch.tensor( # output y
                [self.data.iloc[index + self.sequence_length + self.horizon - 1][self.target_var]],
                dtype=torch.float)
        )

# and a make_loaders function for each horizon h, and each batch_size during tuning
def make_loaders(train_df, train_val_df, h, batch_size,
                 sequence_length, target_var, test_size, num_workers):
    """
    place holder. add docstring later if I feel needed
    """
    train_dataset = SlidingFixedWindow(train_df,
                                       sequence_length = sequence_length,
                                       target_var = target_var,
                                       horizon = h)
    
    val_dataset = SlidingFixedWindow(train_val_df[-(sequence_length + test_size):],
                                    sequence_length = sequence_length,
                                    target_var=target_var,
                                    horizon=h)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
            # training
            learning_rate = 1e-3,
            weight_decay = 0.0,
            loss_name : str = "mae",
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
def tune_with_optuna(train_df : pd.DataFrame,
                     train_val_df: pd.DataFrame,
                     h: int,
                     max_epochs_tune: int,
                     n_trials: int,
                     random_seed: int,
                     sequence_length: int,
                     target_var: str,
                     test_size: int,
                     num_workers: int,):
    """
    Args:
        train_df: training data DataFrame
        train_val_df: training + validation data DataFrame
        h: forecasting horizon
        max_epochs_tune: maximum epochs for tuning
        n_trials: number of optuna trials
        random_seed: random seed for reproducibility
        sequence_length: length of input sequences
        target_var: target variable column name in string
        test_size: size of the test set
        num_workers: number of workers for DataLoader
    
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
        loss_name     = trial.suggest_categorical("loss", ["mae", "huber"])

        # load data using the suggested batch_size
        train_loader, val_loader = make_loaders(train_df, train_val_df, h, batch_size,
                                                sequence_length, target_var, test_size, num_workers)

        model = LSTM_model(
            input_dim=train_val_df.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_name=loss_name,   # change to loss_name later
        )

        callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ]

        trainer = L.Trainer(
        max_epochs=max_epochs_tune,
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
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    return study.best_params

# ==========================================================
# Refit the model on entire train/val dataset with best hyperparameters
# ==========================================================
# output below is a tuple of (best_model, best_path)
def refit_best_model(train_val_df : pd.DataFrame,
                     h: int,
                     best_hps: dict,
                     sequence_length: int,
                     target_var: str,
                     max_epochs_refit: int,
                     random_seed: int,
    ):

    seed_everything(random_seed, workers=True)
    
    # the final dataset for training the best model
    final_df = SlidingFixedWindow(train_val_df, sequence_length, target_var, horizon=h)
    final_loader = DataLoader(final_df, batch_size=best_hps["batch_size"], shuffle=True)

    model = LSTM_model(
        input_dim=train_val_df.shape[1],
        output_dim=1,
        hidden_dim=best_hps["hidden_dim"],
        num_layers=best_hps["num_layers"],
        dropout=best_hps["dropout"],
        learning_rate=best_hps["learning_rate"],
        weight_decay = best_hps.get("weight_decay", 0.0) if best_hps.get("use_weight_decay", False) else 0.0,
        # loss_name="mae",
        loss_name=best_hps["loss"],   # leave for later when I tune loss function
    )

    # save the last checkpoint (no val_loader here because it is a refit)
    ckpt = ModelCheckpoint(
        save_top_k=1,
        monitor=None, # nothing ot monitor because no val_loader
        save_last=True, # keep last.ckpt
        filename=f"last-h{h}" + "-{epoch:03d}",
    )

    final_trainer = L.Trainer(
        max_epochs=max_epochs_refit,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ckpt # no early stopping during refit
        ],
        deterministic=True,
        gradient_clip_val=best_hps.get("grad_clip", 1.0), # default to 1 if not found
        enable_checkpointing=True,
        logger=False,
        log_every_n_steps=10,
    )
    final_trainer.fit(model, final_loader)

    best_path = ckpt.last_model_path # because it's not tuning, just get last.ckpt
    best_model = LSTM_model.load_from_checkpoint(best_path, **model.hparams)
    # **model.hparams not a necessary step but just to be sure I have the right hyperparameters
    best_model.eval()
    return best_model, best_path

# ==========================================================
# This version: fixed model rolling forecast on test set
# ==========================================================
def forecast_on_test(df_h : pd.DataFrame,
                     test_df: pd.DataFrame,
                     model : LSTM_model,
                     h: int,
                     sequence_length: int,
                     target_var: str,):
    """
    Args:
        df_h: the entire dataset (DF)
        test_df: test data (DF)
        model: best LSTNet model from refitting
        h: forecasting horizon
        sequence_length: length of input sequences
        target_var: target variable column name in string
    """
    model.eval() # set to eval mode
    device = next(model.parameters()).device # get model device (cpu or gpu)

    preds = [] # initialte a result container

    with torch.no_grad():
        test_size = test_df.shape[0]
        for idx in range(test_size):

            X = torch.tensor(
                df_h.iloc[-(sequence_length + test_size) -h+1 + idx : - test_size -h+1 + idx].values,
                dtype=torch.float
                ).unsqueeze(0).to(device) # unsqueeze to add batch dimension and move to device
            
            yhat = model(X)

            preds.append(yhat.item()) # get scalar value and append to list
    
    preds = torch.tensor(preds) # convert a list of multiple tensors to a single tensor

    final_out = pd.DataFrame(
        {
            f"{target_var}-pred-h{h}": np.asarray(preds), # predicted values converted to numpy array
            f"{target_var}-true-h{h}": np.asarray(test_df[target_var])
        },
        index=test_df.index
    )
    final_out.index.name = "Date"
    return final_out

# ==========================================================
# Main function to run all horizons
# ==========================================================
def run(config_path="my_config.yaml"):

    # load config
    cfg = build_config_lstm(config_path)

    data = pd.read_csv(cfg.data_file, index_col=0, parse_dates=True)

    df, train_val_data, train_data, val_data, test_data = split_train_test(data,
                                                                       cfg.manual_cut_off_date,
                                                                       cfg.test_size,
    )

    max_h = max(cfg.forecasting_horizon)
    sequence_length = cfg.sequence_length_factor * max_h

    seed_everything(cfg.random_seed, workers=True)

    all_out = []
    for h in cfg.forecasting_horizon:
        best_hps = tune_with_optuna(train_df = train_data,
                                    train_val_df = train_val_data,
                                    h = int(h),
                                    max_epochs_tune = cfg.max_epochs_tune,
                                    n_trials = cfg.n_trials,
                                    random_seed = cfg.random_seed,
                                    sequence_length = sequence_length,
                                    target_var = cfg.target_var,
                                    test_size = cfg.test_size,
                                    num_workers = cfg.num_workers,
        )
        
        model, _ = refit_best_model(train_val_df = train_val_data,
                                    h = int(h),
                                    best_hps = best_hps,
                                    sequence_length = sequence_length,
                                    target_var = cfg.target_var,
                                    max_epochs_refit = cfg.max_epochs_refit,
                                    random_seed = cfg.random_seed,
        )
        
        out_h = forecast_on_test(df_h = df,
                                 test_df = test_data,
                                 model = model,
                                 h = int(h),
                                 sequence_length = sequence_length,
                                 target_var = cfg.target_var,
        )
        
        out_path = cfg.result_root / f"{cfg.model_name}_predictions_h{h}.csv"
        out_h.to_csv(out_path, index=True)
        all_out.append(out_h)
    
    combined = pd.concat(all_out, axis=1)
    combined_path = cfg.result_root / f"{cfg.model_name}_predictions_all_horizons.csv"
    combined.to_csv(combined_path, index=True)

    return {"model": cfg.model_name, "combined_csv": str(combined_path), "result_dir": str(cfg.result_root)}

if __name__ == "__main__":
    run()  
