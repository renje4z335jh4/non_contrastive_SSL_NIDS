import torch
from torch import nn
from abc import abstractmethod

class BaseModel(nn.Module):
    """Base class for all models
    from https://github.com/ireydiak/anomaly_detection_NRCAN
    """

    def __init__(self, in_features: int, n_instances: int, device: str = None, **kwargs):
        super(BaseModel, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_instances = n_instances
        self.in_features = in_features

    @staticmethod
    @abstractmethod
    def load_from_ckpt(model_params: dict, model_state_dict: dict):
        """Loads model with given model state

        Parameters
        ----------
        model_params : dict
            parameter of the model
        model_state_dict : dict
            state of the model
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return dictionary with model parameter

        Returns
        -------
        dict
            model parameter
        """
        return {
            "n_instances": self.n_instances,
            "in_features": self.in_features,
        }

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """This function compute the output of the network in the forward pass

        Parameters
        ----------
        batch : torch.Tensor
            Features of a batch

        Returns
        -------
        torch.Tensor
        """
        pass
