import numpy as np
import torch
import torch.nn.functional as F

from models.ssl.base import BaseRepresentation

class VICReg(BaseRepresentation):
    name = 'VICReg'

    def __init__(
            self,
            in_features: int,
            n_instances: int,
            batch_size: int,
            encoder_class: str,
            mlp_params: dict,
            device: str = None,
            mixup_alpha: float = None,
            sim_coeff: float = 25.0, # Invariance regularization loss coefficient
            std_coeff: float = 25.0, # Variance regularization loss coefficient
            cov_coeff: float = 1.0,  # Covariance regularization loss coefficient
            **encoder_args
        ):
        super().__init__(
            in_features,
            n_instances,
            encoder_class=encoder_class,
            mlp_params=mlp_params,
            device=device,
            **encoder_args
        )
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.mixup_alpha = mixup_alpha
        self.batch_size = batch_size
        self.encoder_args = encoder_args

        self.backbone = self.encoder

    def get_params(self) -> dict:
        base_params = super().get_params()
        del base_params['mlp_params']
        params = {
            'batch_size': self.batch_size,
            'sim_coeff': self.sim_coeff,
            'std_coeff': self.std_coeff,
            'cov_coeff': self.cov_coeff,
            'mixup_alpha': self.mixup_alpha,
            'mlp_params': self.mlp_params
        }
        return dict(
            **base_params,
            **params
        )

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict):
        model = VICReg(**model_params)
        model.load_state_dict(state_dict)
        return model

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.backbone(v1), start_dim=1)
        y = torch.flatten(self.backbone(v2), start_dim=1)

        if self.mixup_alpha:
            # select random sample from batch and perform mixup
            idx = np.random.randint(x.shape[0], size=x.shape[0])
            x = torch.mul(self.mixup_alpha, x[:]) + torch.mul((1 - self.mixup_alpha), x[idx])

            idx = np.random.randint(y.shape[0], size=y.shape[0])
            y = torch.mul(self.mixup_alpha, y[:]) + torch.mul((1 - self.mixup_alpha), y[idx])

        x = self.projector(x)
        y = self.projector(y)

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.in_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.in_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_encoder(self):
        return self.backbone
