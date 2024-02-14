from __future__ import annotations
from typing import Tuple
import torch

from trainer.base import BaseTrainer
from baselines.model.reconstruction import AutoEncoder

class AutoEncoderTrainer(BaseTrainer):

    @staticmethod
    def load_from_file(fname: str, device: str = None) -> Tuple[AutoEncoderTrainer, AutoEncoder]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = AutoEncoder.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = AutoEncoderTrainer(
            model=model,
            **ckpt["trainer_params"]
        )
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # set state
        trainer.metric_values = ckpt["metric_values"]

        return trainer, model

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum(axis=1)

    def train_iter(self, X):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        reg = self.model.reg
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z

        return loss
