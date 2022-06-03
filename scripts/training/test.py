import os
import warnings

import hydra
import omegaconf
import pytorch_lightning as pl
from loguru import logger

import scripts.training.lightning_modules as lm
from polyphonic_music_modeling.model import LSTM as LSTM


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="train_experiment"
)
@logger.catch
def main(configs: omegaconf.DictConfig) -> None:
    logger.add("testing.log")
    logger.info("Creating dataset")

    dataset_module = lm.DataModule(
        batch_size=1,
        num_workers=configs.dataset_module.num_workers,
        dataset_root=configs.dataset_module.dataset_root,
    )

    logger.info("Loading model")
    nn_module = lm.ModelModule.load_from_checkpoint(configs.test.path)

    logger.info("Creating trainer")
    nn_trainer = pl.Trainer(
        max_epochs=configs.training.max_epochs,
        gpus=1 if configs.training.with_gpu else 0,
    )

    nn_trainer.test(nn_module, dataset_module)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
