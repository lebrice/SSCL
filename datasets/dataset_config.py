from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

from simple_parsing import choice, field
from utils.json_utils import Serializable
from .data_utils import keep_in_memory

@dataclass(frozen=True)
class DatasetConfig:
    """
    Represents all the configuration options related to a Dataset.
    """
    # which dataset class to use.
    dataset_class: Type[VisionDataset] = MNIST
    x_shape: Tuple[int, int, int] = (1, 28, 28)
    num_classes: int = 10
    # Transforms to apply to the data
    transforms: Optional[Callable] = ToTensor()
    target_transforms: Optional[Callable] = None
    # Wether we want to load the dataset to memory.
    keep_in_memory: bool = True

    def load(self, data_dir: Path) -> Tuple[Dataset, Dataset]:
        """ Downloads the corresponding train & test datasets and returns them.
        """
        # Use the data_dir argument if given, otherwise use "./data"
        train = self.dataset_class(data_dir, train=True,  download=True, transform=self.transforms)
        test  = self.dataset_class(data_dir, train=False, download=True, transform=self.transforms)
        
        keep_in_memory(train)
        keep_in_memory(test)
        return train, test