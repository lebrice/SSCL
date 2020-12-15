""" Defines the IIDSetting, as a variant of the TaskIncremental setting with
only one task.
"""
from dataclasses import dataclass
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)
import itertools
import tqdm
from torch import Tensor

from sequoia.common.loss import Loss
from sequoia.common.metrics import Metrics
from sequoia.common.config import Config
from sequoia.settings.base import Results
from sequoia.utils.utils import constant, dict_union

from .. import TaskIncrementalSetting
from .iid_results import IIDResults

# TODO: Remove the task labels here.

# TODO: IDEA: Add the pytorch lightning datamodules in the list of
# 'available datasets' for the IID setting, and make sure that it doesn't mess
# up the methods in the parents (train/val loop, dataloader construction, etc.)
# IDEA: Maybe overwrite the 'train/val/test_dataloader' methods on the setting
# and when the chosen dataset is a LightnignDataModule, then just return the
# result from the corresponding method on the LightningDataModule, rather than
# from super().
# from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
#                                   ImagenetDataModule, MNISTDataModule)


@dataclass
class IIDSetting(TaskIncrementalSetting):
    """Your 'usual' learning Setting, where the samples are i.i.d.
    
    Implemented as a variant of Task-Incremental CL, but with only one task.
    
    """
    Results: ClassVar[Type[Results]] = IIDResults

    # Held constant, since this is an IID setting.
    nb_tasks: int = constant(1)
    increment: Union[int, List[int]] = constant(None)
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = constant(None)
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = constant(None)
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = constant(None)
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = constant(None)
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = constant(None)


SettingType = TypeVar("SettingType", bound=IIDSetting)

if __name__ == "__main__":
    IIDSetting.main()
