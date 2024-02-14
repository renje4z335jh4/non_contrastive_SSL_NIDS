from __future__ import annotations
from typing import Union
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
import logging

class KMeans_Eval(KMeans):

    def __init__(
            self,
            encoder = None,
            batch_size: int = 1024,
            device = None,
            **kwargs
        ) -> None:
        super().__init__(n_clusters=1, n_init='auto', **kwargs)
        self.encoder = encoder
        self.batch_size = batch_size

        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def fit(self, X: np.ndarray | DataLoader) -> None:
        logging.info("Fitting started")
        if self.encoder:
            X = self._encode_dataset(X)

        super().fit(X)

    def score(self, X: torch.Tensor) -> np.ndarray:

        if self.encoder:
            X = self._encode_batch(X)
        elif isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        return np.linalg.norm((X - self.cluster_centers_), axis=1)

    def test(self, dataset: DataLoader | np.ndarray, y=None) -> Union[np.array, np.array]:
        if y is not None:
            # model is running on raw data, not onto a representation
            return np.array(y), self.score(dataset)

        y_true, scores = [], []
        with torch.no_grad():
            for batch in dataset:
                x_batch, y_batch = batch

                if type(x_batch) is torch.Tensor:
                    x_batch = x_batch.to(self.device).float()
                else:
                    # list of subsets
                    x_batch = [x.to(self.device).float() for x in x_batch]

                score = self.score(x_batch)
                y_true.extend(y_batch.cpu().tolist())
                scores.extend(score.tolist())

        return np.array(y_true), np.array(scores)

    def validate(self, test_ldr: DataLoader | np.ndarray, y=None):

        logging.info("Evaluation started")

        y_true, scores = self.test(test_ldr, y)
        res = compute_metrics(y_true, scores)

        return {x: res[x] for x in res.keys() if x != 'Confusion Matrix'}


    def _encode_dataset(self, X: np.ndarray | DataLoader) -> np.ndarray:

        assert self.encoder, 'Encoder is not defined'

        if type(X) is np.ndarray:
            data_ldr = DataLoader(X, batch_size=self.batch_size, shuffle=False)
        else:
            data_ldr = X

        x_representation = None
        for batch in data_ldr:
            x, y = batch

            if type(x) is torch.Tensor:
                    x = x.to(self.device).float()
            else:
                # list of subsets
                x = [x_view.to(self.device).float() for x_view in x]

            repr = self._encode_batch(x)

            x_representation = repr if x_representation is None else np.vstack((x_representation, repr))

        return x_representation

    def _encode_batch(self, X: torch.Tensor) -> np.ndarray:

        assert self.encoder, 'Encoder is not defined'

        with torch.no_grad():

            # create list with single item for structure consistency
            if type(X) is torch.Tensor:
                X = [X]
            with torch.no_grad():
                latents = [self.encoder(x).flatten(start_dim=1) for x in X]

                x_representation = torch.mean(torch.stack(latents, dim=0), dim=0)

            x_representation = x_representation.cpu().numpy()

        return x_representation
