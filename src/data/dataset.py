from __future__ import annotations
import logging
from typing import Tuple
import abc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder

from data.augmentation import MultiViewDataInjector

class AbstractDataset(Dataset):
    """Abstract class for data set handling"""

    def __init__(self, path: str):
        super(AbstractDataset, self).__init__()

        self.path = path

        self.features = np.ndarray(0)
        self.labels = np.ndarray(0)

        self.n_instances = 0
        self.in_features = 0

        self.load_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.int] | Tuple[np.ndarray, np.ndarray, np.int]:
        return self.features[index], self.labels[index]

    @abc.abstractmethod
    def load_data(self):
        pass

    def split_train_test(self, test_ratio: float = 0.5, contamination_rate: float = 0.0, pos_label: int = 1) -> Tuple[Subset, Subset]:

        # Fetch and shuffle indices of the majority/normal class
        normal_idx = np.where(self.labels != pos_label)[0]
        normal_shuffled_idx = torch.randperm(len(normal_idx)).long()

        # split majority class samples in test and train
        num_normal_test = int(len(normal_shuffled_idx) * test_ratio)
        normal_test_idx = normal_idx[normal_shuffled_idx[:num_normal_test]]
        normal_train_idx = normal_idx[normal_shuffled_idx[num_normal_test:]]

        # Fetch and shuffle indices of the minority/anomaly class
        anomaly_idx = np.where(self.labels == pos_label)[0]
        anomaly_shuffled_idx = torch.randperm(len(anomaly_idx)).long()

        logging.info("Number normal samples: %i", len(normal_idx))
        logging.info("Number anomaly samples: %i", len(anomaly_idx))

        if contamination_rate > 0:
            # compute number of samples to inject into train set to achieve [contamination_rate]
            num_anomalies_to_inject = int(len(normal_train_idx) * contamination_rate / (1 - contamination_rate))

            assert num_anomalies_to_inject <= len(anomaly_shuffled_idx)

            anomaly_train_idx = anomaly_idx[anomaly_shuffled_idx[:num_anomalies_to_inject]]
            anomaly_test_idx = anomaly_idx[anomaly_shuffled_idx[num_anomalies_to_inject:]]

        else:
            anomaly_train_idx = np.array([], dtype=int)
            anomaly_test_idx = anomaly_idx

        logging.info("Train set contains %i normal samples, %i anomaly samples", len(normal_train_idx), len(anomaly_train_idx))
        logging.info("Test set contains %i normal samples, %i anomaly samples", len(normal_test_idx), len(anomaly_test_idx))

        train_idx = np.concatenate([normal_train_idx, anomaly_train_idx])
        test_idx = np.concatenate([normal_test_idx, anomaly_test_idx])

        if type(self) is MultiViewIntrusionData or type(self) is MultiViewIntrusionDataTransformer:
            return CustomSubset(Subset(self, train_idx), self.train_transform), CustomSubset(Subset(self, test_idx), self.test_transform)
        else:
            return Subset(self, train_idx), Subset(self, test_idx)

    def shrink_dataset(self, normal_size: float) -> None:
        assert normal_size <=1, 'Only a value between 0 < normal_size <=1 is valid'

        indices = np.random.randint(len(self), size=int(len(self) * normal_size))
        self.features = self.features[indices]
        self.labels = self.labels[indices]

class IntrusionData(AbstractDataset):
    def __init__(self, path: str):
        super().__init__(path)

    def load_data(self) -> None:
        if self.path.endswith(".csv"):
            dataset = pd.read_csv(self.path, index_col=False, header=0)
        else:
            raise ValueError("File type not supported")

        self.columns = dataset.columns
        self.features = dataset.values[:, :-1]
        self.labels = dataset.values[:, -1]

        self.n_instances = self.features.shape[0]
        self.in_features = self.features.shape[1]

class IntrusionDataTransformer(AbstractDataset):
    def __init__(self, path: str):
        super().__init__(path)

    def load_data(self) -> None:
        if self.path.endswith(".csv"):
            dataset_df = pd.read_csv(self.path, index_col=False, header=0)
        else:
            raise Exception("File type not supported")

        labels = dataset_df.pop('label')

        self.columns = dataset_df.columns

        # identify categorical columns
        self.categorical_cols = dataset_df.select_dtypes(exclude=np.number).columns.to_list()
        self.categorical_cols_idx = [dataset_df.columns.get_loc(c) for c in self.categorical_cols]

        # identify numeric columns
        self.numeric_cols = dataset_df.select_dtypes(include=np.number).columns.to_list()
        self.numeric_cols_idx = [dataset_df.columns.get_loc(c) for c in self.numeric_cols]

        # encode categorical columns
        for c in self.categorical_cols:
            dataset_df[c] = LabelEncoder().fit_transform(dataset_df[c])

        # save unique (encoded) values for each category
        self.unique_cats = [np.unique(dataset_df[c]) for c in self.categorical_cols]

        self.features = dataset_df.values
        self.labels = np.asarray(labels)

        self.n_instances = self.features.shape[0]
        self.in_features = self.features.shape[1]

class MultiViewIntrusionData(IntrusionData):
    def __init__(
        self,
        path: str,
        train_transform: MultiViewDataInjector = None,
        test_transform: MultiViewDataInjector = None,
        shuffle_features = False
    ):
        super().__init__(path)
        self.shuffle_features = shuffle_features

        # suffle data set features
        if self.shuffle_features:
            rnd_perm = np.random.permutation(self.features.shape[1])
            self.features = self.features[:, rnd_perm]
            self.columns = [self.columns[i] for i in rnd_perm]
            logging.info("Features were shuffled. When loading a model, the exact same order must be reconstructed.")
            logging.info("Feature order: " + ', '.join(self.columns))

        assert test_transform is None or (test_transform is not None and train_transform is not None), 'Transforming the test set is only necessary for subset augmentation.'

        self.train_transform = train_transform
        if not self.train_transform is None:
            self.train_transform.generate_subset_properties(self.in_features)

        self.test_transform = test_transform
        if not test_transform is None:
            self.test_transform.generate_subset_properties(self.in_features)

        self.in_features = self.train_transform.n_features_subset + self.train_transform.n_overlap if self.train_transform else self.features.shape[1]


class MultiViewIntrusionDataTransformer(IntrusionData):
    def __init__(
        self,
        path: str,
        train_transform: MultiViewDataInjector = None,
        test_transform: MultiViewDataInjector = None,
        shuffle_features = False
    ):
        self.shuffle_features = shuffle_features
        super().__init__(path)

        assert test_transform is None or (test_transform is not None and train_transform is not None), 'Transforming the test set is only necessary for subset augmentation.'

        self.train_transform = train_transform
        if not self.train_transform is None:
            self.train_transform.generate_subset_properties(self.in_features)

        self.test_transform = test_transform
        if not test_transform is None:
            self.test_transform.generate_subset_properties(self.in_features)

        self.in_features = self.train_transform.n_features_subset + self.train_transform.n_overlap

    def load_data(self) -> None:
        if self.path.endswith(".csv"):
            dataset_df = pd.read_csv(self.path, index_col=False, header=0)
        else:
            raise Exception("File type not supported")

        labels = dataset_df.pop('label')

        # suffle data set features
        if self.shuffle_features:
            dataset_df = dataset_df.reindex(columns=np.random.permutation(dataset_df.columns))
            logging.info("Features were shuffled. When loading a model, the exact same order must be reconstructed.")
            logging.info("Feature order: " + ', '.join(dataset_df.columns))

        self.columns = dataset_df.columns

        # identify categorical columns
        self.categorical_cols = dataset_df.select_dtypes(exclude=np.number).columns.to_list()
        self.categorical_cols_idx = [dataset_df.columns.get_loc(c) for c in self.categorical_cols]

        # identify numeric columns
        self.numeric_cols = dataset_df.select_dtypes(include=np.number).columns.to_list()
        self.numeric_cols_idx = [dataset_df.columns.get_loc(c) for c in self.numeric_cols]

        # encode categorical columns
        for c in self.categorical_cols:
            dataset_df[c] = LabelEncoder().fit_transform(dataset_df[c])

        # save unique (encoded) values for each category
        self.unique_cats = [np.unique(dataset_df[c]) for c in self.categorical_cols]

        self.features = dataset_df.values
        self.labels = np.asarray(labels)

        self.n_instances = self.features.shape[0]
        self.in_features = self.features.shape[1]

class CustomSubset(Dataset):
    """Custom subset to provide different transformations for test and train set
    """

    def __init__(self, subset: Subset, transform: MultiViewDataInjector = None):
        super().__init__()

        self.features, self.labels = subset.dataset[subset.indices]
        self.dataset = subset
        self.transform = transform

        self.num_features = subset.dataset.in_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform:

            random_shuffled_sample = None
            # get param for ShuffleSwapNoise
            if self.transform.ssn_used:
                # generate a sample with position consistent feature sampling
                random_shuffled_sample = np.diagonal(self.dataset[np.random.randint(len(self),size=len(feature))][0])

            return self.transform(
                    feature,
                    random_shuffled_sample=random_shuffled_sample,
                ), self.labels[index]
        else:
            return feature, self.labels[index]
