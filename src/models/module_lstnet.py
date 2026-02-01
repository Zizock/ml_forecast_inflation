# ==========================================================
# A callable LSTNet module for time series forecasting
# ==========================================================
# This script does the following:
# define an LSTNet layer,
# for a given back testing length,
# for each forecasting horizon,
# tune and train an LSTNet model with data from train/val period,
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
import torch.nn.functional as F
import torch.nn as nn

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
class config_lstnet:
    model_name: str
    result_root: pathlib.Path
    data_file: pathlib.Path
    random_seed: int

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

# ==== build config dataclass for LSTNet ====
def build_config_lstnet(config_path : pathlib.Path) -> config_lstnet:
    my_config = load_config(config_path)
    model_name = "lstnet"
    result_root = repo_path("results", model_name)
    result_root.mkdir(parents=True, exist_ok=True)

    return config_lstnet(
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
# Create a LSTNet model class in nn.Module
# ==========================================================
# describe LSTNet architecture class in nn.Module, used later in lightning
class LSTNetModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        sequence_length,

        # attention (when skip == 0)
        num_attn_heads = 4,
        hidden_state_dims_attn = 32,

        # CNN + GRU
        n_out_channels = 10,
        window_size = 2,
        hidden_state_dims_GRU1 = 32,
        skip = 4,
        hidden_state_dims_GRU2 = 32,
        dropout = 0.0,

        # target variable column position
        target_idx= 0,
    ):
        super().__init__()

        self.target_idx = target_idx
        self.skip = skip
        self.sequence_length = sequence_length

        # conv: (batch, 1, seq, input_dim) -> (batch, n_out_channels, seq-window+1, 1)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_out_channels,
            kernel_size=(window_size, input_dim),
        )

        # dropout rate
        self.dropout = nn.Dropout(dropout)

        # GRU1 on conv outputs
        self.GRU1 = nn.GRU(
            input_size=n_out_channels,
            hidden_size=hidden_state_dims_GRU1,
            batch_first=True,
            dropout=(dropout if hidden_state_dims_GRU1 > 1 else 0.0), # put it here in case I add more later
        )

        # GRU2 for skip branch
        self.GRU2 = nn.GRU(
            input_size=n_out_channels,
            hidden_size=hidden_state_dims_GRU2,
            batch_first=True,
            dropout=(dropout if hidden_state_dims_GRU2 > 1 else 0.0), # put it here in case I add more later
        )

        # head for recurrent features
        self.linear1 = nn.Linear(hidden_state_dims_GRU1 + skip * hidden_state_dims_GRU2, output_dim)

        # head for attention branch (when skip == 0)
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_state_dims_attn,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1_attn = nn.Linear(hidden_state_dims_attn + hidden_state_dims_GRU1, output_dim)

        # AR head (only the target variable)
        self.linear2 = nn.Linear(sequence_length, 1)

    def forward(self, inputs):
        # inputs: (batch, sequence_length, input_dim), so make sure to use batch_first=True in RNNs
        batch_size = inputs.shape[0]

        # (1) CNN part
        h_conv = F.relu(self.conv1(inputs.unsqueeze(1)).squeeze(-1)) # (batch, ch, T',)
        h_conv = self.dropout(h_conv)

        # (2) GRU1 part
        x_gru = h_conv.permute(0, 2, 1) # change dimension order to (batch, T', ch)
        H_gru, h_gru = self.GRU1(x_gru) # H_gru: (batch, T', hid1), h_gru: (1,batch,hid1)
        # note: h_gru is the last hidden state
        h_gru = h_gru.squeeze(0) # remove the num_layers dimension because it is 1
        H_gru, h_gru = self.dropout(H_gru), self.dropout(h_gru)

        # (3)-(4) skip GRU2 OR attention branch, depending on skip value
        if self.skip > 0:
            # (3) Recurrent-skip Component (GRU for every p hidden states) GRU2
            seq_len = x_gru.shape[1] // self.skip # each sequence will have these many elements
            n_seq = self.skip # there will be these many sequences
            
            c = x_gru[:, -int(seq_len * n_seq):]  # (batch, seq_len*n_seq, ch), discard the states which can't fit in the window
            c = c.view(batch_size, seq_len, n_seq, c.shape[-1]).contiguous() # stride every n_seq before switching index
            c = c.permute(0, 2, 1, 3).contiguous().view(batch_size * n_seq, seq_len, c.shape[-1]) # switch the dimensions and obtain the input for GRU2
            
            _, s = self.GRU2(c) # s: (1, batch*n_seq, hid2)
            s = self.dropout(s)

            # (4) recurrent skip component (concatenation)
            # this is the concatenation of the last hidden states from both GRUs
            r = torch.cat((h_gru, s.view(batch_size, -1)), dim=1)
            res = self.linear1(r)  # (batch, 1), linear mapping to output
        else: # when skip == 0, use attention to capture long-term dependencies
            # attention uses H_gru embedding space
            attn_out, _ = self.attn_layer(query=H_gru[:, -1:], key=H_gru, value=H_gru)  # (batch,1,attn_dim)
            r2 = torch.cat((h_gru, attn_out.squeeze(1)), dim=1)
            res = self.linear1_attn(r2)

        # (5) AR on target variable only
        x_target = inputs[:, :, self.target_idx] # (batch, sequence_length)
        z = self.linear2(x_target) # (batch, 1)

        # (6) final output: additive of the two parts
        return res + z
    
# ==========================================================
# Build LSTNet model using PyTorch Lightning
# ==========================================================
class LSTNet_Lightning(L.LightningModule):
    """
    Args:
        input_dim (int): Number of time series df.shape[1]
        output_dim (int): number of time series to predict, in my case, 1 (only inflation)
        sequence_length (int): number of past time steps given as input
        
        n_out_channels (int): number of kernels in the convolution layer
        window_size (int): kernel width in convolution layer
        hidden_state_dims_GRU1 (int): dimension of the first recurrent component 
        skip (int): number of hidden units to skip in skip-recurrent component
        hidden_state_dims_GRU2 (int): dimension of the second GRU unit (skip-recurrent component)
        num_attn_heads (int): number of attention heads in the attention unit (used only if skip > 0)
        hidden_state_dims_attn (int): dimension of attention layer (used only if skip > 0)
        dropout (float): probability of dropout (similar to regularization )
        target_idx (int): index of the target variable in the input data

        learning_rate (float): starting learning rate for Adam optimizer
        weight_decay (float): weight decay for AdamW optimizer
        loss_name (str): loss function to use, either "mae" or "huber" (I use mae here, maybe try huber later)
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        sequence_length,
        # model
        n_out_channels = 16,
        window_size = 2,
        hidden_state_dims_GRU1 = 64,
        skip = 4,
        hidden_state_dims_GRU2 = 64,
        num_attn_heads = 4,
        hidden_state_dims_attn = 64,
        dropout = 0.0,
        target_idx = 0,
        # training
        learning_rate = 1e-3,
        weight_decay = 0.0,
        loss_name : str = "mae",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.LSTNet = LSTNetModel(
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,

            num_attn_heads=num_attn_heads,
            hidden_state_dims_attn=hidden_state_dims_attn,

            n_out_channels=n_out_channels,
            window_size=window_size,
            hidden_state_dims_GRU1=hidden_state_dims_GRU1,
            skip=skip,
            hidden_state_dims_GRU2=hidden_state_dims_GRU2,
            dropout=dropout,

            target_idx=target_idx,
        )

        if loss_name == "huber": # not activated yet
            self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        else:
            self.loss_fn = nn.L1Loss()

    def forward(self, X):
        return self.LSTNet(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X)
        loss = self.loss_fn(yhat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        yhat = self(X)
        loss = self.loss_fn(yhat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # added AdamW and try a w_d=0, which is equivalent to Adam
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

# ==========================================================
# Define training and tuning with optuna
# ==========================================================
def tune_with_optuna(train_df : pd.DataFrame,
                     train_val_df : pd.DataFrame,
                     h : int,
                     target_idx : int,
                     max_epochs_tune : int,
                     n_trials : int,
                     random_seed : int,
                     sequence_length : int,
                     target_var : str,
                     test_size : int,
                     num_workers : int,):
    """
    Args:
        train_df: training data DataFrame
        train_val_df: training + validation data DataFrame
        h: forecasting horizon
        target_idx: index of the target variable in the input data (used for AR component)
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
        # training hyperparams
        batch_size    = trial.suggest_categorical("batch_size", [8, 12, 16, 20, 24])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)

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
        
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)
        loss_name = trial.suggest_categorical("loss", ["mae", "huber"])

        # architecture hyperparams
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        skip    = trial.suggest_categorical("skip", [0, 2, 3, 4, 6])

        n_out_channels         = trial.suggest_categorical("n_out_channels", [8, 16, 32, 64])
        window_size            = trial.suggest_categorical("window_size", [2, 3, 4, 6])
        hidden_state_dims_GRU1 = trial.suggest_categorical("hidden_state_dims_GRU1", [16, 32, 64, 128])
        hidden_state_dims_GRU2 = trial.suggest_categorical("hidden_state_dims_GRU2", [8, 16, 32, 64])

        # attention only used when skip==0; embed_dim must be divisible by heads
        if skip == 0:
            num_attn_heads = trial.suggest_categorical("num_attn_heads", [1, 2, 4, 8])
            hidden_state_dims_attn = hidden_state_dims_GRU1  # tie to GRU1 dim
            if hidden_state_dims_attn % num_attn_heads != 0:
                raise optuna.TrialPruned()
        else:
            num_attn_heads = 4
            hidden_state_dims_attn = hidden_state_dims_GRU1

        train_loader, val_loader = make_loaders(train_df, train_val_df, h, batch_size,
                                                sequence_length, target_var, test_size, num_workers)

        model = LSTNet_Lightning(
            input_dim=train_df.shape[1],
            output_dim=1,
            sequence_length=sequence_length,
            
            n_out_channels=n_out_channels,
            window_size=window_size,
            hidden_state_dims_GRU1=hidden_state_dims_GRU1,
            skip=skip,
            hidden_state_dims_GRU2=hidden_state_dims_GRU2,
            num_attn_heads=num_attn_heads,
            hidden_state_dims_attn=hidden_state_dims_attn,
            dropout=dropout,
            target_idx=target_idx,

            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_name=loss_name,
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=False),
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
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
                     target_idx: int,
                     sequence_length: int,
                     target_var: str,
                     max_epochs_refit: int,
                     random_seed: int,):

    seed_everything(random_seed, workers=True)

    # the final dataset for training the best model
    final_df = SlidingFixedWindow(train_val_df, sequence_length, target_var, horizon=h)
    final_loader = DataLoader(final_df, batch_size=best_hps["batch_size"], shuffle=True)

    # attention parameters
    skip = best_hps["skip"]
    hidden_state_dims_GRU1 = best_hps["hidden_state_dims_GRU1"]
    if skip == 0:
        num_attn_heads = best_hps.get("num_attn_heads", 4)
        hidden_state_dims_attn = hidden_state_dims_GRU1
    else:
        num_attn_heads = 4
        hidden_state_dims_attn = hidden_state_dims_GRU1

    model = LSTNet_Lightning(
        input_dim=train_val_df.shape[1],
        output_dim=1,
        sequence_length=sequence_length,

        n_out_channels=best_hps["n_out_channels"],
        window_size=best_hps["window_size"],
        hidden_state_dims_GRU1=hidden_state_dims_GRU1,
        skip=skip,
        hidden_state_dims_GRU2=best_hps["hidden_state_dims_GRU2"],
        num_attn_heads=num_attn_heads,
        hidden_state_dims_attn=hidden_state_dims_attn,
        dropout=best_hps["dropout"],
        target_idx=target_idx,
        learning_rate=best_hps["learning_rate"],
        weight_decay = best_hps.get("weight_decay", 0.0) if best_hps.get("use_weight_decay", False) else 0.0,
        loss_name=best_hps["loss"],
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
        gradient_clip_val=best_hps.get("grad_clip", 1.0),
        enable_checkpointing=True,
        logger=True,
        log_every_n_steps=10,
    )

    # # try later
    # # another version: monitor train_loss and use early stopping
    # ckpt = ModelCheckpoint(
    #     monitor="train_loss",
    #     mode="min",
    #     save_top_k=1,
    #     filename=f"best-train-h{h}" + "-{epoch:03d}-{train_loss:.4f}",
    # )

    # final_trainer = L.Trainer(
    #     max_epochs=max_epochs_refit,
    #     accelerator="auto",
    #     devices="auto",
    #     callbacks=[
    #         ckpt,
    #         EarlyStopping(monitor="train_loss", mode="min", patience=20, verbose=True),
    #     ],
    #     deterministic=True,
    #     gradient_clip_val=best_hps.get("grad_clip", 1.0),
    #     enable_checkpointing=True,
    #     logger=True,
    #     log_every_n_steps=10,
    # )

    final_trainer.fit(model, final_loader) # final training doesn't need a val loader

    best_path = ckpt.last_model_path
    best_model = LSTNet_Lightning.load_from_checkpoint(best_path, **model.hparams)
    # **model.hparams not a necessary step but just to be sure I have the right hyperparameters
    best_model.eval()
    return best_model, best_path

# ==========================================================
# This version: fixed model rolling forecast on test set
# ==========================================================
def forecast_on_test(df_h : pd.DataFrame,
                     test_df: pd.DataFrame,
                     model : LSTNet_Lightning,
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
    model.eval()
    device = next(model.parameters()).device

    preds = [] # initiate a result container

    with torch.no_grad():
        test_size = test_df.shape[0]
        for idx in range(test_size):
            X = torch.tensor(
                df_h.iloc[
                    -(sequence_length + test_size) -h + 1 + idx : -test_size - h + 1 + idx
                    ].values,
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
    cfg = build_config_lstnet(config_path)

    data = pd.read_csv(cfg.data_file, index_col=0, parse_dates=True)

    df, train_val_data, train_data, val_data, test_data = split_train_test(data,
                                                                       cfg.manual_cut_off_date,
                                                                       cfg.test_size,
    )
    
    max_h = max(cfg.forecasting_horizon)
    sequence_length = cfg.sequence_length_factor * max_h
    target_idx = df.columns.get_loc(cfg.target_var)

    seed_everything(cfg.random_seed, workers=True)

    all_out = []
    for h in cfg.forecasting_horizon:
        best_hps = tune_with_optuna(train_df = train_data,
                                    train_val_df = train_val_data,
                                    h = int(h),
                                    target_idx = target_idx,
                                    sequence_length = sequence_length,
                                    target_var = cfg.target_var,
                                    random_seed= cfg.random_seed,
                                    n_trials= cfg.n_trials,
                                    max_epochs_tune = cfg.max_epochs_tune,
                                    num_workers= cfg.num_workers,
                                    test_size= cfg.test_size,
        )

        model, _ = refit_best_model(train_val_df = train_val_data,
                                    h = int(h),
                                    target_idx = target_idx,
                                    sequence_length = sequence_length,
                                    target_var = cfg.target_var,
                                    random_seed= cfg.random_seed,
                                    max_epochs_refit = cfg.max_epochs_refit,
                                    best_hps = best_hps,
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