from __future__ import annotations
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
import torch

from ..model.one_class import DeepSVDD
from trainer.base import BaseTrainer


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, R=None, c=None, **kwargs):
        super(DeepSVDDTrainer, self).__init__(**kwargs)
        self.c = c

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = DeepSVDD.load_from_ckpt(ckpt)
        trainer = DeepSVDDTrainer(model=model, c=ckpt["c"], batch_size=1, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    @staticmethod
    def load_from_file(fname: str, device: str = None) -> Tuple[DeepSVDDTrainer, DeepSVDD]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = DeepSVDD.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = DeepSVDDTrainer(
            model=model,
            **ckpt["trainer_params"]
        )
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # set state
        trainer.metric_values = ckpt["metric_values"]

        return trainer, model

    def train_iter(self, sample: torch.Tensor):
        assert torch.allclose(self.c, torch.zeros_like(self.c)) is False, "center c not initialized"
        outputs = self.model(sample)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        return torch.mean(dist)

    def score(self, sample: torch.Tensor):
        assert torch.allclose(self.c, torch.zeros_like(self.c)) is False, "center c not initialized"
        outputs = self.model(sample)
        return torch.sum((outputs - self.c) ** 2, dim=1)

    def before_training(self, dataset: DataLoader):
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(dataset)
            print("Center c initialized.")

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _ = sample
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        if torch.allclose(c, torch.zeros_like(c)):
            raise Exception("Center c initialized at 0")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.model.train(mode=True)
        return c

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "c": self.c,
        }
