"""Implements backbone of BYOL algorithm
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import numpy as np
from models.ssl.encoder import BaseEncoder
from models.ssl.base import BaseRepresentation
from models.ssl.mlp import MLP

class BYOL(BaseRepresentation):
    """BYOL model
    """
    name = 'BYOL'

    def __init__(
        self,
        in_features: int,
        n_instances: int,
        encoder_class: str,
        mlp_params: dict,
        target_decay_rate: float,
        device: str = None,
        mixup_alpha: float = None,
        **encoder_args
    ):
        super(BYOL, self).__init__(in_features, n_instances, encoder_class, mlp_params, device, **encoder_args)

        self.target_decay_rate = target_decay_rate
        self.mixup_alpha = mixup_alpha

        self.online_network = SiameseArm(
            self.EncoderClass,
            mlp_params=self.mlp_params,
            mixup_alpha=mixup_alpha,
            **encoder_args
        )
        with torch.no_grad():
            self.target_network = copy.deepcopy(self.online_network)
            self.target_network.mixup_alpha = mixup_alpha

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict) -> BYOL:
        model = BYOL(**model_params)
        model.load_state_dict(state_dict)
        return model

    def get_params(self) -> dict:
        base_params = super().get_params()
        params = {
            'target_decay_rate': self.target_decay_rate,
            'mixup_alpha': self.mixup_alpha,
        }
        return {**base_params, **params}

    def forward(self, sample):
        return super().forward(sample)

    def get_encoder(self) -> BaseEncoder:
        return self.online_network.encoder

    @torch.no_grad()
    def update_target_network(self):
        for online_param, target_param in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target_param.data = target_param.data * self.target_decay_rate + online_param.data * (1. - self.target_decay_rate)

class SiameseArm(nn.Module):
    def __init__(
        self,
        EncoderClass: BaseEncoder,
        mlp_params: dict,
        mixup_alpha: float = None,
        **encoder_args
    ):
        super(SiameseArm, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.encoder = EncoderClass(**encoder_args)
        encoder_out_dim = self.encoder.output_dim

        self.projector = MLP(
            input_dim=encoder_out_dim,
            **mlp_params
        )
        self.predictor = MLP(
            input_dim=mlp_params['output_dim'],
            **mlp_params
        )

    def forward(self, inputs):

        embedding = self.encoder(inputs)

        if self.mixup_alpha:
            # select random sample from batch and perform mixup
            idx = np.random.randint(inputs.shape[0], size=inputs.shape[0])
            embedding = torch.mul(self.mixup_alpha, embedding[:]) + torch.mul((1 - self.mixup_alpha), embedding[idx])

        projection = self.projector(embedding)

        prediction = self.predictor(projection)

        return embedding, projection, prediction
