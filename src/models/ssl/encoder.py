"""Implemented encoder
"""
import torch
import torch.nn as nn
from abc import abstractmethod
from models.ssl.mlp import MLP

class BaseEncoder(nn.Module):

    def __init__(self, output_dim: int) -> None:
        super(BaseEncoder, self).__init__()

        self.output_dim = output_dim

    def forward(self, batch: torch.Tensor):
        pass

    def get_output_dim(self) -> int:
        """Returns the output dimension of the encoder

        Returns
        -------
        int
            output dimension
        """
        return self.output_dim


class CNN(BaseEncoder):
    # similar to https://arxiv.org/pdf/2209.03147.pdf

    def __init__(self, num_features: int):
        output_dim = 512 * int((int((int((num_features - 3) / 3) - 1) / 2) - 1) / 4)
        remove_last_pooling, remove_last_conv = False, False
        if output_dim == 0:
            # remove last pooling layer
            print('Last pooling layer is removed to fit the input shape')
            remove_last_pooling = True
            output_dim = 512 * (int((int((num_features - 3) / 3) - 1) / 2) -1)
        if output_dim == 0:
            # remove last conv layer
            print('Last conv layer is removed to fit the input shape')
            remove_last_conv = True
            output_dim = 256 * int((int((num_features - 3) / 3) - 1) / 2)

        super(CNN, self).__init__(output_dim)

        kernel_size = (1, 2)

        layers = [
            nn.Conv2d(1, 32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(128, 256, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size),
        ]
        if not remove_last_conv:
            layers.append(nn.Conv2d(256, 512, kernel_size=kernel_size)),
            layers.append(nn.ReLU()),
        if not remove_last_pooling:
            layers.append(nn.MaxPool2d(kernel_size=(1, 4)))
        layers.append(nn.Flatten(1))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):

        # adding one dimension (channel) for IDS data -> 'dummy' dimension
        inputs = inputs.unsqueeze(dim=-1)

        # reshape to 2D with height = 1
        inputs = inputs.unsqueeze(dim=1)

        # [B,H,W,C] -> [B,C,H,W]
        inputs = inputs.permute(0, 3, 1, 2)

        inputs = self.model(inputs)

        return inputs

class MLP_Encoder(BaseEncoder):
    output_dim = 256

    def __init__(
        self,
        num_features: int,
    ):
        super(MLP_Encoder, self).__init__(MLP_Encoder.output_dim)

        self.num_features = num_features

        # MLP encoder parameter from SCARF implementation
        embedding_dim = 256
        batch_norm = True
        dropout = None
        encoder_depth = 4

        self.model = MLP(
            input_dim=self.num_features,
            embedding_dim=embedding_dim,
            output_dim=MLP_Encoder.output_dim,
            n_layers=encoder_depth,
            batch_norm=batch_norm,
            dropout=dropout,
        ).model

    def forward(self, inputs):
        inputs = torch.flatten(inputs, 1)
        inputs = self.model(inputs)
        return inputs
