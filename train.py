import wandb
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    # CALLBACK
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename="best.ckpt", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    # WanDB Logger
    wandb_logger = WandbLogger(project="ops-basics", entity="taehee-k", name="bert")

    # bring together DataLoader and LightningModule
    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,  # run one bactch of training -> run one batch of validation
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        deterministic=True,
        # limit_on_train_batches=0.25,
        # limit_on_val_batches=0.25,
    )

    # TRAINING
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
