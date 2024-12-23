import itertools
import torch
import math
from torch.utils.data.sampler import Sampler
from util.misc import all_gather, shared_random_seed
from util import misc
from collections import defaultdict

class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None, len_dataset=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        self._epoch = 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part
        self.num_samples = int(math.ceil(len(self._int_part) * 1.0 / self._world_size))
        self.total_size = self.num_samples * self._world_size
        self.len_dataset = len_dataset

    def __len__(self) -> int:
        return len(self.indices)
        # return len(self._frac_part // self._world_size)
    
    def set_epoch(self, epoch):
        self._epoch = epoch
    
    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed+self._epoch)
        indices = self._get_epoch_indices(g)
        if self._shuffle:
            randperm = torch.randperm(len(indices), generator=g)
            indices = indices[randperm].tolist()
        else:
            indices = indices.tolist()
        # import pdb; pdb.set_trace()
        # if self.len_dataset is not None:
        #     indices = indices[:self.len_dataset]

        # print(len(indices), self.total_size, self._rank, self._world_size, self.num_samples)
        if len(indices) % self._world_size:
            indices += indices[:(self._world_size -len(indices) % self._world_size)]

        assert len(indices) % self._world_size == 0
        self.indices = indices[self._rank::self._world_size]

        return iter(self.indices)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()

from typing import Optional
class ClassAwareSampler(Sampler):
    def __init__(self, dataset_dicts, seed: Optional[int] = None, sample_size=120000):
        """
        """
        # self._size = len(dataset_dicts)
        self._size = sample_size
        self._epoch = 0
        assert self._size > 0
        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)

        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()
        self.weights = self._get_class_balance_factor(dataset_dicts)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self._size // self._world_size

    def __iter__(self):
        start = self._rank
        g = torch.Generator()
        g.manual_seed(self._seed+self._epoch)
        ids = torch.multinomial(self.weights, self._size, generator=g, replacement=True)
        return itertools.islice(
            ids, start, None, self._world_size)

    def _get_class_balance_factor(self, dataset_dicts, l=1.):
        ret = []
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for i, dataset_dict in enumerate(dataset_dicts):
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            ret.append(sum(
                [1. / (category_freq[cat_id] ** l) for cat_id in cat_ids]))
        return torch.tensor(ret).float()