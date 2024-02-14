"""Trainer for BYOL
"""
from __future__ import annotations
import itertools
from typing import List, Tuple
import torch

from trainer.base import BaseTrainer
from models.ssl.byol import BYOL

class BYOL_Trainer(BaseTrainer):
    def __init__(
        self,
        model: BYOL,
        batch_size,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        n_epochs: int = 200,
        n_jobs_dataloader: int = 0,
        device: str = "cuda",
        anomaly_label=1,
        ckpt_root: str = None,
        test_ldr=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            n_jobs_dataloader=n_jobs_dataloader,
            device=device,
            anomaly_label=anomaly_label,
            ckpt_root=ckpt_root,
            test_ldr=test_ldr,
            **kwargs
        )

    @staticmethod
    def load_from_file(fname: str, device: str = None) -> Tuple[BYOL_Trainer, BYOL]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = BYOL.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = BYOL_Trainer(
            model=model,
            **ckpt["trainer_params"]
        )
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # set state
        trainer.metric_values = ckpt["metric_values"]
        trainer.loss_values = ckpt["loss_values"]

        return trainer, model

    def train_iter(self, views) -> torch.Tensor:

        #view1, view2 = views
        combinations = self.get_combinations_of_subsets(views)

        train_losses = []
        for view1, view2 in combinations:

            # compute loss for view1 to view2
            y1, z1, h1 = self.model.online_network(view1)
            with torch.no_grad():
                y2, z2, h2 = self.model.target_network(view2)

            loss1 = regression_loss(h1, z2.detach())

            # compute loss for view2 to view1
            y1, z1, h1 = self.model.online_network(view2)
            with torch.no_grad():
                y2, z2, h2 = self.model.target_network(view1)

            loss2 = regression_loss(h1, z2.detach())

            train_losses.append(torch.mean(loss1 + loss2))

        # final loss
        train_loss = torch.mean(torch.stack(train_losses))

        return train_loss

    def after_batch(self):
        self.model.update_target_network()

    def score(self, sample: torch.Tensor):
        pass

    def after_training(self) -> None:
        pass

def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # normalization of each element in x, y
    normed_x, normed_y = _l2_normalize(x, axis=-1), _l2_normalize(y, axis=-1)

    return torch.sum((normed_x - normed_y) ** 2, axis=-1)


def _l2_normalize(x: torch.Tensor, axis: int = None) -> torch.Tensor:
    """l2 normalize a tensor on an axis with numerical stability."""
    square_sum = torch.sum(
        torch.square(x), axis=axis, keepdims=True
    )  # sum(elementwise_square (x_i*x_i)) = |x| for each element in x
    x_inv_norm = torch.rsqrt(
        square_sum  # , epsilon , epsilon: float = 1e-12
    )  # numerical save -> if square_sum < epsilon take epsilon
    return x * x_inv_norm
