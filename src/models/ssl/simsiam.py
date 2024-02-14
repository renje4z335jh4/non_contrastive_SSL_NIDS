import numpy as np
import torch
import torch.nn as nn

from models.ssl.base import BaseRepresentation
from models.ssl.mlp import MLP

class SimSiam(BaseRepresentation):
    name = 'SimSiam'

    def __init__(
        self,
        in_features: int,
        n_instances: int,
        encoder_class: str,
        mlp_params: dict,
        mixup_alpha: float = None,
        device: str = None,
        **encoder_args
    ):
        super().__init__(in_features, n_instances, encoder_class, mlp_params, device, **encoder_args)
        self.mixup_alpha = mixup_alpha
        self.predictor = MLP(input_dim=mlp_params['output_dim'], **self.mlp_params) # identical to the projector

    def get_params(self) -> dict:
        params = {
            'mixup_alpha': self.mixup_alpha
        }
        return {**super().get_params(), **params}

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict):
        model = SimSiam(**model_params)
        model.load_state_dict(state_dict)
        return model

    def get_encoder(self):
        return self.encoder

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        if self.mixup_alpha:
            # select random sample from batch and perform mixup
            idx = np.random.randint(z1.shape[0], size=z1.shape[0])
            z1 = torch.mul(self.mixup_alpha, z1[:]) + torch.mul((1 - self.mixup_alpha), z1[idx])

            idx = np.random.randint(z2.shape[0], size=z2.shape[0])
            z2 = torch.mul(self.mixup_alpha, z2[:]) + torch.mul((1 - self.mixup_alpha), z2[idx])

        # projection
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        # prediction
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
