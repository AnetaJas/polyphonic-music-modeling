import logging
import os
import warnings
import datetime 

import hydra
import omegaconf
import pytorch_lightning as pl

import scripts.training.lightning_modules as lm
import scripts.dataset.dataset_loading as ds
from polyphonic_music_modeling.model import LSTM as LSTM
from pytorch_lightning.loggers import WandbLogger


from loguru import logger

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
    #dataset_module.setup()
    
    logger.info("Creating model")
    nn_module = lm.ModelModule(
        lr=configs.nn_module.lr,
        model= LSTM(input_dim=88) # do korelacji ze zbiore danych 
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
    )


    nn_trainer.fit(nn_module, dataset_module)
    
    nn_trainer.test(nn_module, dataset_module)
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
    
