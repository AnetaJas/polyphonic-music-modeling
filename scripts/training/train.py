import datetime
import logging
import os
import sys
import warnings
import datetime 

import hydra
import omegaconf
import pytorch_lightning as pl

sys.path.append(os.getcwd())

from loguru import logger
from pytorch_lightning.loggers import WandbLogger

import scripts.dataset.dataset_loading as ds
import scripts.training.lightning_modules as lm
from polyphonic_music_modeling.model import LSTM as LSTM
from pytorch_lightning.loggers import WandbLogger



@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="train_experiment"
)
@logger.catch
def main(configs: omegaconf.DictConfig) -> None:
    logger.add("training.log")
    logger.info("Creating dataset")

    dataset_module = lm.DataModule(
        batch_size=configs.dataset_module.batch_size,
        num_workers=configs.dataset_module.num_workers,
        dataset_root=configs.dataset_module.dataset_root
    )
    # dataset_module.setup()

    logger.info("Creating model")
    nn_module = lm.ModelModule(
        lr=configs.nn_module.lr,
        model=LSTM(input_dim=88),  # do korelacji ze zbiore danych
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandblogger = WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
        log_model="all",
    )
    
    logger.info("📲 Initializing callbacks.")


    model_ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor=configs.training.early_stop.monitor,
        mode=configs.training.early_stop.mode,
        # fmt: off
        filename=configs.training.wandb_name + "-{epoch}-{" + configs.training.early_stop.monitor + ":.4f}",
        # fmt: on
        save_top_k=3,
        dirpath="./models",
        save_last=True,
    )
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandblogger = WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
        log_model="all" 
    )
    
    logger.info("Creating trainer")
    nn_trainer = pl.Trainer(
        max_epochs=configs.training.max_epochs,
        logger=wandblogger,
        gpus=1 if configs.training.with_gpu else 0,
        callbacks=[model_ckpt_callback],
        log_every_n_steps=1
    )

    nn_trainer.fit(nn_module, dataset_module)
    
    logger.info(f"🥇 Best model: {model_ckpt_callback.best_model_path}")
    
    nn_module.load_from_checkpoint(model_ckpt_callback.best_model_path)
    
    nn_trainer.test(nn_module, dataset_module)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
