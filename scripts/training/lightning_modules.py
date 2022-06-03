import os
import typing as t

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.dataset.dataset_loading import MusicDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
        dataset_root: str,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root

    def setup(self, stage: t.Optional[str] = None) -> None:

        self.train_dataset = MusicDataset(
            dataset_root=os.path.join(self.dataset_root, "train")
        )

        self.validation_dataset = MusicDataset(
            dataset_root=os.path.join(self.dataset_root, "val")
        )

        self.test_dataset = MusicDataset(
            dataset_root=os.path.join(self.dataset_root, "test")
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def ret_dim(self):
        return self.train_dataset.ret_dimension()


class ModelModule(pl.LightningModule):
    def __init__(self, *, lr: float, model: nn.Module):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = lr
        self.criterion = nn.NLLLoss()
        self.neural_net = model

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.neural_net.parameters(), lr=self.learning_rate)

    def training_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch=batch)

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(log_prefix="train", outputs=outputs)

    def validation_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch=batch)

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(log_prefix="val", outputs=outputs)
        pass

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(log_prefix="test", outputs=outputs)

    def test_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch=batch)

    def _step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor]
    ) -> pl.utilities.types.STEP_OUTPUT:
        x, y = batch
        y_pred = self.neural_net(x)
        loss = self.criterion(y_pred, y[:][0]) # to fit dimension of batch data
        return {"loss": loss}

    def _summarize_epoch(
        self, log_prefix: str, outputs: pl.utilities.types.EPOCH_OUTPUT
    ):
        mean_loss = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"{log_prefix}_loss", mean_loss, on_epoch=True)
