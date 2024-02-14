"""Data augmentations
"""
import math
from typing import List, Dict, Union
from random import randint, random, shuffle
import numpy as np
import abc

class BaseAugmentation(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    abc.abstractmethod
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """Converts the sample to another view of the sample

        Parameters
        ----------
        sample : np.ndarray
            sample to augment

        Returns
        -------
        np.ndarray
            augmented sample
        """

class FisherYates(BaseAugmentation):
    def __call__(self, sample: np.ndarray) -> np.ndarray:
         # save shape
        shape = sample.shape

        # flat 2D to 1D
        sample = sample.flatten()

        max_index = len(sample) - 1
        for i in range(max_index, 1, -1):
            j = randint(0, i)

            sample[i], sample[j] = sample[j], sample[i]

        return sample.reshape(shape)

class RandomHorizontalFlip(BaseAugmentation):
    def __init__(self, p: float=0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if random() > self.p:
            return np.fliplr(sample)
        else:
            return sample

class RandomVerticalFlip(BaseAugmentation):
    def __init__(self, p: float=0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if random() > self.p:
            return np.flipud(sample)
        else:
            return sample

class RandomCrop(BaseAugmentation):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __call__(self, sample: np.ndarray) -> np.ndarray:

        orig_shape = sample.shape
        if orig_shape[0] <= self.size:
            raise ValueError('size must be smaller than the shape of the sample')

        start_i_idx = np.random.randint(0, orig_shape[0] - self.size + 1)
        start_j_idx = np.random.randint(0, orig_shape[0] - self.size + 1)

        return sample[start_i_idx:start_i_idx + self.size, start_j_idx:start_j_idx + self.size]

class NoiseBase(BaseAugmentation):
    """Base class for different noise/masking augmentations
    """
    def __init__(self, p: float, mode: str = 'bernoulli') -> None:
        """Base class for noise/masking augmentations

        Parameters
        ----------
        p : float
            probability for mode 'bernoulli', or ratio of features for mode 'ratio'
        mode : str, optional
            mode to create the mask. Either sample from Bernoulli distribution
            or set fixed ratio, by default 'bernoulli'
        """
        super().__init__()
        self.p = p # masking ratio for mode=ratio, probability for mode=bernoulli

        assert mode in ['bernoulli', 'ratio'], '[mode] must be one of "bernoulli", "ratio"'
        self.mode = mode

    def generate_mask(self, sample_len: int) -> np.ndarray:

        if self.mode == 'ratio':
            #number of values to mask
            n_mask_values = int(sample_len * self.p)

            # shuffle indices and take first n_mask_values indices
            mask_idx = np.random.permutation(sample_len)[:n_mask_values]

            mask = np.zeros(sample_len, dtype=np.int8)
            mask[mask_idx] = 1
        else:
            mask = np.random.binomial(1, self.p, sample_len)

        return mask

class GaussianNoise(NoiseBase):
    """Adds gaussian sampled noise on *p* features of the sample
    """
    def __init__(self, p: float=0.5, mode: str = 'bernoulli', mean: float = 0.0, std: float = 0.1) -> None:
        super().__init__(p, mode)
        self.std = std
        self.mean = mean

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        # get mask
        mask = self.generate_mask(len(sample))

        # create noise
        noise = np.random.normal(self.mean, self.std, sample.shape)

        return np.clip(np.where(mask, sample+noise, sample), a_min=0, a_max=1)

class ZeroOutNoise(NoiseBase):
    """Replace *p* features of the sample with 0
    """
    def __init__(self, p: float=0.5, mode: str = 'bernoulli') -> None:
        super().__init__(p, mode)

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        # get mask
        mask = self.generate_mask(len(sample))

        # create zero vector
        zeros = np.zeros(sample.shape)

        return np.where(mask, zeros, sample)

class ShuffleSwapNoise(NoiseBase):
    """ Augmentation where *p* percentage features of the input sample will be
        exchanged with the *same* features of other *samples* from dataset.
    """
    def __init__(self, p: float=0.5, mode: str = 'bernoulli') -> None:
        super().__init__(p, mode)

    def __call__(self, sample: np.ndarray, random_shuffled_sample: np.ndarray) -> np.ndarray:
        assert len(sample) == len(random_shuffled_sample), 'Must have same size'
        # get mask
        mask = self.generate_mask(len(sample))

        # where mask == 1 take feature from random_sample
        # where mask == 0 take feature from sample
        return np.where(mask, random_shuffled_sample, sample)

class Compose():
    """Compose of augmentations
    """
    def __init__(self, *functions) -> None:
        self.functions = functions

    def __call__(
            self,
            sample: np.ndarray,
            random_shuffled_sample: np.ndarray = None,
            random_sample: np.ndarray = None,
        ) -> np.ndarray:

        for func in self.functions:
            if type(func) is ShuffleSwapNoise:
                sample = func(sample, random_shuffled_sample)
            else:
                sample = func(sample)

        return sample.copy()

class MultiViewDataInjector(object):
    def __init__(
            self,
            transformations: List[List[dict]],
            n_subsets: int = 2,
            overlap: float = 1.0,
            training: bool = True
        ):
        """Generator for sampling of data set.
        Generates n_subsets subsets/views of one sample.
        In case of the defaults and no transformations, 2 copies of the
        sample will be returned.

        Parameters
        ----------
        transformations : List[List[dict]]
            List containing a list of transformations (dict).
            One list of transformations for each subset.
        n_subsets : int, optional
            Number of subsets to generate, by default 2
        overlap : float, optional
            Percentage of overlap between subsets, by default 1.0
        training : bool, optional
            Whether in train mode: shuffles the subsets to avoid bias, by default True
        """

        self.ssn_used = False

        assert n_subsets >= 2, 'At least 2 subsets must be created'
        assert overlap >= 0.0 and overlap <= 1.0, 'invalid overlap, must between 0 <= overlap <= 1'
        assert len(transformations) == n_subsets, 'Number of subsets must match the number of given transformations'

        self.n_subsets = n_subsets
        self.overlap = overlap
        self.training = training

        self.transformations = [self.create_transformations(t) for t in transformations]

    def __call__(self, sample, random_shuffled_sample: np.ndarray):
        subsets = self.subset_generator(sample)

        # generate also subsets of the random samples for following augmentations
        # else create list of None's
        if random_shuffled_sample is None:
            random_shuffled_sample = [None for _ in range(self.n_subsets)]
        else:
            random_shuffled_sample = self.subset_generator(random_shuffled_sample)

        if not self.transformations is None:
            subsets = [transform(sub, random_shuffled_sample=rs_sample) if not transform is None else sub for sub, transform, rs_sample in zip(subsets, self.transformations, random_shuffled_sample)]

        # shuffle order of subsets to avoid any bias during training. The order is unchanged at the test time.
        if self.training:
            shuffle(subsets)

        return subsets

    def create_transformations(self, transform_parameter: Union[List[Dict], None]) -> Compose:
        """Generates composition class of transformations

        Parameters
        ----------
        transform_parameter : Union[List[Dict], None]
            List of dictionary, where each dictionary contains information and
            parameter of the transformation.

            example: [{'ShuffleSwapNoise': {'p': 0.5}}, {'ZeroOutNoise': {'p': 0.5, 'mode': 'bernoulli'}}]

        Returns
        -------
        Compose
            Composition of transformations
        """

        if transform_parameter is None:
            return None

        list_of_transforms = []
        # for each transformation of view_i
        for transform_func in transform_parameter:
            # create the transformation with given parameter
            # transform_func only contains ONE key
            for transform_name, transform_params in transform_func.items():

                # save if ShuffleSwapNoise or SwapNoise is used -> need samples as parameter
                if transform_name == 'ShuffleSwapNoise':
                    self.ssn_used = True

                list_of_transforms.append(globals()[transform_name](**transform_params if not transform_params is None else {}))

        return Compose(*list_of_transforms)

    def generate_subset_properties(self, n_features: int) -> None:
        """Generates a list of tuples containing start and stop (column)
        indices for each subset.

        Parameters
        ----------
        n_features : int
            number of features in data set (alias columns)
        """
        self.n_features_subset = math.ceil(n_features / self.n_subsets)
        # Number of overlapping features between subsets
        self.n_overlap = min(int(self.overlap * self.n_features_subset), int(n_features / self.n_subsets))

        subset_feature_idx = []

        # generate indices for each subset
        for i in range(self.n_subsets):
            start_idx = max(0, i * self.n_features_subset - self.n_overlap)
            stop_idx = max(self.n_features_subset + self.n_overlap, (i + 1) * self.n_features_subset)

            # corner case: number of features not dividable by n_features_subset+n_overlap
            # move last subset to cover remaining features
            # overlap decreases, but size of subset remains
            if i == self.n_subsets - 1 and stop_idx != n_features :
                diff = n_features - stop_idx
                stop_idx += diff
                start_idx += diff

            subset_feature_idx.append((start_idx, stop_idx))

        self.subset_feature_idx = subset_feature_idx

    def subset_generator(self, sample: np.ndarray) -> List[np.ndarray]:
        """Generates subsets of the sample.

        Parameters
        ----------
        sample : np.ndarray
            Original sample of the data set

        Returns
        -------
        List[np.ndarray]
            List of subsets of the sample
        """

        # generate subsets
        subsets = []
        for start_idx, stop_idx in self.subset_feature_idx:
            subsets.append(sample[start_idx:stop_idx])

        return subsets
