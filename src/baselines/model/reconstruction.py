from __future__ import annotations

from models.base import BaseModel
from ..model import utils
from ..model.utils import activation_mapper

class AutoEncoder(BaseModel):
    """
    Implements a basic Deep Auto Encoder
    """
    name = "AE"

    def __init__(
            self,
            compression_factor: int,
            latent_dim: int = 2,
            act_fn: str = 'relu',
            n_layers: int = 4,
            reg: float = 0.5,
            **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.act_fn = activation_mapper[act_fn]
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.encoder, self.decoder = None, None
        self.reg = reg
        self._build_network()

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict) -> AutoEncoder:
        model = AutoEncoder(**model_params)
        model.load_state_dict(state_dict)
        return model

    def _build_network(self):
        # Create the ENCODER layers
        enc_layers = []
        in_features = self.in_features
        compression_factor = self.compression_factor
        for _ in range(self.n_layers - 1):
            out_features = in_features // compression_factor
            enc_layers.append(
                [in_features, out_features, self.act_fn]
            )
            in_features = out_features
        enc_layers.append(
            [in_features, self.latent_dim, None]
        )
        # Create DECODER layers by simply reversing the encoder
        dec_layers = [[b, a, c] for a, b, c in reversed(enc_layers)]
        # Add and remove activation function from the first and last layer
        dec_layers[0][-1] = self.act_fn
        dec_layers[-1][-1] = None
        # Create networks
        self.encoder = utils.create_network(enc_layers)
        self.decoder = utils.create_network(dec_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        output = self.encoder(x)
        output = self.decoder(output)
        return x, output

    def get_params(self) -> dict:
        params = {
            "latent_dim": self.latent_dim,
            "act_fn": str(self.act_fn).lower().replace("()", ""),
            "n_layers": self.n_layers,
            "compression_factor": self.compression_factor,
            "reg": self.reg
        }

        return dict(
            super().get_params(),
            **params
        )
