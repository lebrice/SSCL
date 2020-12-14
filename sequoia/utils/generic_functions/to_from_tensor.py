from functools import singledispatch
from typing import Any, Dict, Union, Tuple, Optional

import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor
from ._namedtuple import NamedTuple


@singledispatch
def from_tensor(space: Space, sample: Union[Tensor, Any]) -> Union[np.ndarray, Any]:
    """ Converts a Tensor into a sample from the given space. """
    if isinstance(sample, Tensor):
        return sample.cpu().numpy()
    return sample


@from_tensor.register
def _(space: spaces.Discrete, sample: Tensor) -> int:
    if isinstance(sample, Tensor):
        return sample.item()
    elif isinstance(sample, np.ndarray):
        assert sample.size == 1
        return int(sample)
    return sample


@from_tensor.register
def _(space: spaces.Dict, sample: Dict[str, Union[Tensor, Any]]) -> Dict[str, Union[np.ndarray, Any]]:
    return {
        key: from_tensor(space[key], value)
        for key, value in sample.items()
    }


@from_tensor.register
def _(space: spaces.Tuple, sample: Tuple[Union[Tensor, Any]]) -> Tuple[Union[np.ndarray, Any]]:
    if not isinstance(sample, tuple):
        # BUG: Sometimes instead of having a sample of Tuple(Discrete(2))
        # be `(1,)`, its `array([1])` instead.
        sample = tuple(sample)
    values_gen = (
        from_tensor(space[i], value)
        for i, value in enumerate(sample)
    )
    if isinstance(sample, NamedTuple):
        return type(sample)(values_gen)
    return tuple(values_gen)


@singledispatch
def to_tensor(space: Space,
              sample: Union[np.ndarray, Any],
              device: torch.device = None) -> Union[np.ndarray, Any]:
    """ Converts a sample from the given space into a Tensor. """
    return torch.as_tensor(sample, device=device)


@to_tensor.register
def _(space: spaces.MultiBinary,
      sample: np.ndarray,
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return torch.as_tensor(sample, device=device, dtype=torch.bool)


@to_tensor.register
def _(space: spaces.Dict,
      sample: Dict[str, Union[np.ndarray, Any]],
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return {
        key: to_tensor(space[key], value, device)
        for key, value in sample.items()
    }


@to_tensor.register
def _(space: spaces.Tuple,
      sample: Tuple[Union[np.ndarray, Any], ...],
      device: torch.device = None) -> Tuple[Union[Tensor, Any], ...]:
    return tuple(
        to_tensor(subspace, sample[i], device)
        for i, subspace in enumerate(space.spaces)
    )
    