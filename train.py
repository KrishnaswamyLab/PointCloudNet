# Training the autoencoder.

#from pytorch_lightning.loggers import LightningLoggerBase
from typing import List, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pcnn import DATA_DIR
import torch
from omegaconf import OmegaConf
import pandas as pd
import time
import wandb

def get_logger(exp_name, wandb_user, wandb_disabled = False):
    logger = WandbLogger(
        name=exp_name,
        project='pcnn',
        entity=wandb_user,
        log_model=False
    )

    if wandb_disabled:
        log_dir = "./tmp"
    else:
        log_dir = logger.experiment.dir
    return logger, log_dir


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if not cfg.trainer.wandb:
        wandb.init(mode="disabled")
        

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)() 
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, input_dim=datamodule.input_dim, pos_dim = datamodule.pos_dim, num_classes = datamodule.num_classes)
    
    model.to(torch.float32)

    logger, log_dir = get_logger(
        exp_name=cfg.name, wandb_user=cfg.trainer.wandb_user, wandb_disabled=not cfg.trainer.wandb)

    # checkpointing
    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', patience=cfg.trainer.early_stopping)

    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, logger=logger, callbacks=[
                         checkpoint_cb, early_stopping_cb], max_epochs=cfg.trainer.max_epochs,
                          gradient_clip_val = cfg.trainer.gradient_clip_val, devices = 1)
    trainer.fit(model, datamodule=datamodule)

    val_metrics = trainer.test(model, dataloaders = [datamodule.val_dataloader()], 
                                ckpt_path = "best")
    
    val_metrics = {k.replace("test","val"): v for k,v in val_metrics[0].items()}
    
    test_metrics = trainer.test(model, dataloaders = [datamodule.test_dataloader()], 
                                ckpt_path = "best")
    
    df_val = pd.DataFrame(val_metrics, index = [0])
    df_test = pd.DataFrame(test_metrics, index = [0])
    df = df_val.join(df_test)
    df.to_pickle(
        f"final_{cfg.dataset_name}_{cfg.model_name}.pkl")

@hydra.main(version_base = None, config_path="config/", config_name="main.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934

    # Train model
    return train(config)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
