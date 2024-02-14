from abc import ABC, abstractmethod
import itertools
import os
import time
from typing import List, Union
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch import optim
# from tqdm import trange
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.metrics import compute_metrics

class BaseTrainer(ABC):
    name = "BaseTrainer"

    def __init__(self,
                 model,
                 batch_size,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 anomaly_label=1,
                 ckpt_root: str = None,
                 test_ldr=None,
                 save_best_model: bool = False,
                 **kwargs):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.anomaly_label = anomaly_label
        self.weight_decay = weight_decay
        self.optimizer = self.set_optimizer(weight_decay=self.weight_decay)
        self.ckpt_root = ckpt_root
        self.test_ldr = test_ldr
        self.metric_values = {"Precision": [], "Recall": [], "F1-Score": [], "AUPR": [], "Accuracy": [], "AUROC": []}
        self.loss_values = []
        self.time_values = []
        self.epoch = kwargs.get('epoch', -1)
        self.save_best_model = save_best_model
        self.best_loss = np.finfo(np.float64).max


    @staticmethod
    @abstractmethod
    def load_from_file(fname: str, device: str = None):
        """Loads model and trainer from file for further training/evaluation

        Parameters
        ----------
        fname : str
            file name of stored data
        device : str, optional
            device to load model/trainer on, by default None
        """

    def save_ckpt(self, fname: str):
        general_params = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_values": self.metric_values,
            "loss_values": self.loss_values,
            "time_values": self.time_values,
        }
        trainer_params = {'trainer_params': self.get_params()}
        model_params = {'model_params': self.model.get_params()}
        torch.save(dict(**general_params, **model_params, **trainer_params), fname)

    def get_params(self) -> dict:
        return {
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "anomaly_label": self.anomaly_label,
            "weight_decay": self.weight_decay,
        }

    @abstractmethod
    def train_iter(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self, weight_decay):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, dataset: DataLoader):
        self.model.train(mode=True)
        self.before_training(dataset)
        assert self.model.training, "Model not in training mode. Aborting"

        print("Started training")
        for epoch in range(self.epoch+1, self.n_epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            self.epoch = epoch
            assert self.model.training, "model not in training mode, aborting"
            # with trange(len(dataset)) as t:
            with tqdm(total=len(dataset)) as t:
                for i_batch, batch in enumerate(dataset):

                    X, _ = BaseTrainer.batch_to_device(batch, self.device)

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / (i_batch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()

            epoch_loss = epoch_loss / len(dataset)
            self.loss_values.append(epoch_loss)
            self.time_values.append(time.time() - epoch_start_time)

            # safe best performing model according to loss
            if self.save_best_model and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_ckpt(
                    os.path.join(self.ckpt_root, "best_model.pt")
                )


            if ((epoch+1) % 5 == 0 or epoch == 0):
                # evalute during training
                if self.test_ldr is not None:
                    self.evaluate(self.test_ldr)

                # save checkpoint
                if self.ckpt_root and not self.save_best_model:
                    self.save_ckpt(
                        os.path.join(self.ckpt_root, "{}_epoch={}.pt".format(self.model.name.lower(), epoch + 1))
                    )

        self.after_training()

    @staticmethod
    def get_combinations_of_subsets(subsets: List[torch.Tensor]):

        # Compute combinations of subsets [(x1, x2), (x1, x3)...]
        subset_combinations = list(itertools.combinations(subsets, 2))

        return subset_combinations

    @staticmethod
    def batch_to_device(batch: torch.Tensor, device: str):
        """Puts all parts of the batch onto the device with
        respect to multiple created views in SSL

        Parameters
        ----------
        batch : torch.Tensor
            Input batch
        device : str
            device to train on
        """

        # extract labels from views
        # 'views' contains 1 or more different views of the features
        (X), y = batch

        # convert tensor to list of tensor
        if not type(X) is list:
            X = [X]

        X = [X.to(device).float() for X in X]
        y = y.to(device).long()

        # unpack list ot single tensor
        if len(X) == 1:
            X, = X

        return X, y

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def evaluate(self, test_ldr):
        """Evaluates the model with the given test loader.
        """
        y_true, scores = self.test(test_ldr)
        self.model.train(mode=True)
        res = compute_metrics(y_true, scores)
        self.metric_values["Precision"].append(res["Precision"])
        self.metric_values["Recall"].append(res["Recall"])
        self.metric_values["F1-Score"].append(res["F1-Score"])
        self.metric_values["AUPR"].append(res["AUPR"])
        self.metric_values["Accuracy"].append(res["Accuracy"])
        self.metric_values["AUROC"].append(res["AUROC"])

        return self.metric_values

    def plot_metrics(self, figname="fig1.png"):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """

        precision, recall = self.metric_values["Precision"], self.metric_values["Recall"]
        f1, aupr = self.metric_values["F1-Score"], self.metric_values["AUPR"]
        ac, auroc = self.metric_values["Accuracy"], self.metric_values["AUROC"]
        epochs = [1] + [x * 5 for x in range(1, len(precision))]

        f, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(
            epochs, precision, '-o', label="Precision", c="b"
        )
        ax1.plot(
            epochs, recall, '-o', label="Recall", c="g"
        )
        ax1.plot(
            epochs, aupr, '-o', label="AUPR", c="c"
        )
        ax1.plot(
            epochs, f1, '-o', label="F1-Score", c="r"
        )
        ax1.plot(
            epochs, auroc, '-o', label="AUROC", c="y"
        )
        ax1.plot(
            epochs, ac, '-o', label="Accuracy", c="m"
        )
        ax1.set_xlabel("Epochs", fontsize=16)
        ax1.set_ylabel("Metrics", fontsize=16)
        ax1.legend(fontsize=14)

        f.savefig(figname)
        plt.show()
