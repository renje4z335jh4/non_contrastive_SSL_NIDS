import torch
import torch.nn as nn

class MLP(nn.Module):
    """Multi Layer Perception. 2 linear layers with BN and RELU.
    For projection and prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        output_dim: int,
        n_layers: int,
        batch_norm: bool,
        dropout: float
    ):
        super(MLP, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim

        input_dim = self.input_dim
        layers = []
        for _ in range(n_layers-1):
            layers.append(nn.Linear(input_dim, embedding_dim, bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(embedding_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))

            input_dim = embedding_dim # set in dim = out dim

        layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, 1)
        inputs = self.model(inputs)
        return inputs
