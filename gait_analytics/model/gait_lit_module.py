import pytorch_lightning as lit
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from numpy.typing import NDArray

from gait_analytics.model.gait_model import GaitNet


class GaitDataset(Dataset):
    def __init__(self, X: NDArray[np.floating], y: NDArray[np.floating]):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class GaitLightingModel(lit.LightningModule):
    def __init__(self, model: GaitNet, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss() # Mean Squared Error is best for speed prediction

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
