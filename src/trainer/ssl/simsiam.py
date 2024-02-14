import torch
import torch.nn as nn

from trainer.base import BaseTrainer
from models.ssl.simsiam import SimSiam

class SimSiam_Trainer(BaseTrainer):

    def __init__(self, model, batch_size, lr: float = 0.0001, weight_decay: float = 0.0001, n_epochs: int = 200, n_jobs_dataloader: int = 0, device: str = "cuda", anomaly_label=1, ckpt_root: str = None, test_ldr=None, save_best_model: bool = False, **kwargs):
        super().__init__(model, batch_size, lr, weight_decay, n_epochs, n_jobs_dataloader, device, anomaly_label, ckpt_root, test_ldr, save_best_model, **kwargs)

        self.criterion = nn.CosineSimilarity(dim=1).to(self.device)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)

        model = SimSiam.load_from_ckpt(ckpt['model_params'], ckpt['model_state_dict'])
        trainer = SimSiam_Trainer(
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

    def train_iter(self, batch: torch.Tensor):

        combinations = self.get_combinations_of_subsets(batch)

        train_losses = []
        for v1, v2 in combinations:

            # compute output and loss
            p1, p2, z1, z2 = self.model(v1, v2)
            train_losses.append(-(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5)

        loss = torch.mean(torch.stack(train_losses))

        return loss
