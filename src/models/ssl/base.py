from abc import abstractmethod
from models.base import BaseModel

from models.ssl.encoder import CNN, MLP_Encoder
from models.ssl.transformer.transformer_encoder import FTTransformerEncoder
from models.ssl.mlp import MLP

class BaseRepresentation(BaseModel):
    """Base class for representation models
    """
    def __init__(
            self,
            in_features: int,
            n_instances: int,
            encoder_class: str,
            mlp_params: dict,
            device: str = None,
            **encoder_args
        ):
        super().__init__(in_features, n_instances, device, **encoder_args)

        self.encoder_class = encoder_class
        self.mlp_params = mlp_params
        self.encoder_args = encoder_args

        self.EncoderClass = globals()[self.encoder_class]
        self.encoder = self.EncoderClass(**self.encoder_args)

        self.projector = MLP(
            input_dim=self.encoder.get_output_dim(),
            **mlp_params
        )

    def get_params(self) -> dict:
        base_params = super().get_params()
        params = {
            'encoder_class': self.encoder_class,
            'mlp_params': self.mlp_params,
            **self.encoder_args
        }
        return {**base_params, **params}

    @abstractmethod
    def get_encoder(self):
        """returns the encoder of the representation model

        Returns
        -------
        BaseEncoder
            trained encoder of representation model
        """
