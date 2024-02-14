from __future__ import annotations
from torch import Tensor
from models.base import BaseModel
from ..model.utils import activation_mapper, create_network


class DeepSVDD(BaseModel):
    name = "DeepSVDD"

    def __init__(
        self,
        compression_factor: int = 2,
        n_layers: int = 4,
        act_fn: str = 'relu',
        **kwargs
    ):
        super(DeepSVDD, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.rep_dim = None
        self.act_fn = activation_mapper[act_fn]
        self._build_network()

    def _build_network(self):
        in_features = self.in_features
        compression_factor = self.compression_factor
        out_features = in_features // compression_factor
        layers = []
        for _ in range(self.n_layers - 1):
            layers.append([in_features, out_features, self.act_fn])
            in_features = out_features
            out_features = in_features // compression_factor
            assert out_features > 0, "out_features {} <= 0".format(out_features)
        layers.append(
            [in_features, out_features, None]
        )
        self.rep_dim = out_features
        self.net = create_network(layers).to(self.device)

    @staticmethod
    def load_from_ckpt(model_params: dict, state_dict: dict) -> DeepSVDD:
        model = DeepSVDD(**model_params)
        model.load_state_dict(state_dict)
        return model

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        params = {
            "n_layers": self.n_layers,
            "compression_factor": self.compression_factor,
            "act_fn": str(self.act_fn).lower().replace("()", "")
        }
        return dict(
            **super().get_params(),
            **params
        )
