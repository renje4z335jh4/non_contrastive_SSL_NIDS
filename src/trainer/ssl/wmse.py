import torch
import torch.nn as nn

from trainer.base import BaseTrainer
from models.ssl.wmse import WMSE

class WMSE_Trainer(BaseTrainer):

    def __init__(self, model, batch_size, lr: float = 0.0001, weight_decay: float = 0.0001, n_epochs: int = 200, n_jobs_dataloader: int = 0, device: str = "cuda", anomaly_label=1, ckpt_root: str = None, test_ldr=None, save_best_model: bool = False, **kwargs):
        super().__init__(
            model,
            batch_size,
            lr,
            weight_decay,
            n_epochs,
            n_jobs_dataloader,
            device,
            anomaly_label,
            ckpt_root,
            test_ldr,
            save_best_model,
            **kwargs
        )

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = WMSE.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = WMSE_Trainer(
            model=model,
            **ckpt["trainer_params"]
        )
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # set state
        trainer.metric_values = ckpt["metric_values"]

        return trainer, model

    def get_params(self) -> dict:
        return super().get_params()

    def score(self, sample: torch.Tensor):
        return super().score(sample)

    def train_iter(self, sample: torch.Tensor):

        loss = self.model(sample)

        return loss
