import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    
    cola_data = DataModule(
        model_name=cfg.model.tokenizer,
        batch_size=cfg.processing.batch_size,
        max_length=cfg.processing.max_length,
    )
    cola_model = ColaModel(model_name=cfg.model.name, lr=cfg.model.lr)

    # CALLBACK
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", filename="best-checkpoint", monitor="valid/loss", mode="min"
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
        logger=wandb_logger,
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=cfg.training.fast_dev_run,  # run one bactch of training -> run one batch of validation
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )

    # TRAINING
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()
