import torch

from trainer.base import BaseTrainer
from models.ssl.barlowtwins import BarlowTwins

class BarlowTwins_Trainer(BaseTrainer):

    def __init__(self, model, batch_size, lr: float = 0.0001, weight_decay: float = 0.0001, n_epochs: int = 200, n_jobs_dataloader: int = 0, device: str = "cuda", anomaly_label=1, ckpt_root: str = None, test_ldr=None, save_best_model: bool = False, **kwargs):
        super().__init__(model, batch_size, lr, weight_decay, n_epochs, n_jobs_dataloader, device, anomaly_label, ckpt_root, test_ldr, save_best_model, **kwargs)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = BarlowTwins.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = BarlowTwins_Trainer(
            model=model,
            **ckpt["trainer_params"]
        )
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # set state
        trainer.metric_values = ckpt["metric_values"]

        return trainer, model

    def save_ckpt(self, fname: str):
        return super().save_ckpt(fname)

    def get_params(self) -> dict:
        return super().get_params()

    def train_iter(self, batch: torch.Tensor):

        combinations = self.get_combinations_of_subsets(batch)

        train_losses = []
        for v1, v2 in combinations:
            train_losses.append(self.model(v1, v2))

        loss = torch.mean(torch.stack(train_losses))

        return loss

    def score(self, sample: torch.Tensor):
        return super().score(sample)
