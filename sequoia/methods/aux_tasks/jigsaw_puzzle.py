from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .auxiliary_task import AuxiliaryTask


class JigsawPuzzleTask(AuxiliaryTask):
    def get_loss(self,
                 x: Tensor,
                 h_x: Tensor,
                 y_pred: Tensor,
                 y: Tensor=None) -> Tensor:
        return torch.zeros(1)
        raise NotImplementedError("TODO")