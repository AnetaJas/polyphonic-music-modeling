import datetime
import os
import sys
import warnings
from aifc import Error

import hydra
import omegaconf
import pytorch_lightning as pl

# sys.path.append(os.getcwd())

from loguru import logger
from pytorch_lightning.loggers import WandbLogger

import scripts.dataset.dataset_loading as ds
import scripts.training.lightning_modules as lm
from polyphonic_music_modeling.model import GRU as GRU
from polyphonic_music_modeling.model import LSTM as LSTM
from polyphonic_music_modeling.model import RT


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="train_experiment"
)
@logger.catch
def main(configs: omegaconf.DictConfig) -> None:
    model_to_choose = {
        "LSTM": LSTM(
            input_dim=88,
            lstm_layers=configs.training.model.layers,
            embedding_dim=configs.training.model.embedding_dim,
            hidden_dim=configs.training.model.hidden_dim,
        ),
        "GRU": GRU(
            input_dim=88,
            gru_layers=configs.training.model.layers,
            embedding_dim=configs.training.model.embedding_dim,
            hidden_dim=configs.training.model.hidden_dim,
        ),
        "R-Transfomfer": RT(
            input_size=88,
            d_model=configs.training.model.d_model,
            output_size=configs.training.model.embedding_dim,
            h=configs.training.model.h,
            rnn_type=configs.training.model.rnn_type,
            ksize=configs.training.model.ksize,
            n_level=configs.training.model.n_level,
            n=configs.training.model.n,
            dropout=configs.training.model.dropout,
            emb_dropout=configs.training.model.dropout,
        ),
    }
    logger.add("training.log")
    logger.info("Creating dataset")

    dataset_module = lm.DataModule(
        batch_size=configs.dataset_module.batch_size,
        num_workers=configs.dataset_module.num_workers,
        dataset_root=configs.dataset_module.dataset_root,
    )

    logger.info("Creating model")
    nn_module = lm.ModelModule(
        lr=configs.nn_module.lr,
        model=model_to_choose[
            configs.training.model_name
        ],  # do korelacji ze zbiore danych
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
        filename=configs.training.wandb_name
        + "-{epoch}-{"
        + configs.training.early_stop.monitor
        + ":.4f}",
        save_top_k=3,
        dirpath="./models",
        save_last=True,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandblogger = WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
        log_model="all",
    )

    logger.info("Creating trainer")
    nn_trainer = pl.Trainer(
        max_epochs=configs.training.max_epochs,
        logger=wandblogger,
        gpus=1 if configs.training.with_gpu else 0,
        callbacks=[model_ckpt_callback],
        log_every_n_steps=1,
    )

    nn_trainer.fit(nn_module, dataset_module)

    logger.info(f"🥇 Best model: {model_ckpt_callback.best_model_path}")

    nn_module.load_from_checkpoint(model_ckpt_callback.best_model_path)

    nn_trainer.test(nn_module, dataset_module)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
