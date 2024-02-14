import numpy as np
import torch
import torch.nn as nn

from models.ssl.base import BaseRepresentation

class BarlowTwins(BaseRepresentation):
    name = 'BarlowTwins'

    def __init__(
            self,
            in_features: int,
            n_instances: int,
            encoder_class: str,
            batch_size: int,
            mlp_params,
            device: str = None,
            lambd: float = 0.0051, # weight on off-diagonal terms
            mixup_alpha: float = None,
            **encoder_args
        ):
        super().__init__(
            in_features,
            n_instances,
            encoder_class,
            mlp_params,
            device,
            **encoder_args
        )
        self.lambd = lambd
        self.mixup_alpha = mixup_alpha
        self.batch_size = batch_size

        self.backbone = self.encoder

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.mlp_params['output_dim'], affine=False)

    def get_params(self) -> dict:
        base_params = super().get_params()
        params = {
            'batch_size': self.batch_size,
            'lambd': self.lambd,
            'mixup_alpha': self.mixup_alpha,
        }
        return dict(
            **base_params,
            **params
        )

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict):
        model = BarlowTwins(**model_params)
        model.load_state_dict(state_dict)
        return model

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)

        if self.mixup_alpha:
            # select random sample from batch and perform mixup
            idx = np.random.randint(z1.shape[0], size=z1.shape[0])
            z1 = torch.mul(self.mixup_alpha, z1[:]) + torch.mul((1 - self.mixup_alpha), z1[idx])

            idx = np.random.randint(z2.shape[0], size=z2.shape[0])
            z2 = torch.mul(self.mixup_alpha, z2[:]) + torch.mul((1 - self.mixup_alpha), z2[idx])

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_encoder(self):
        return self.backbone
