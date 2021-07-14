from functools import singledispatch, wraps
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor

from sequoia.common.spaces.image import Image, ImageTensorSpace
from sequoia.common.spaces.named_tuple import NamedTupleSpace
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.utils.generic_functions import from_tensor, move, to_tensor
from sequoia.utils.logging_utils import get_logger
from collections import abc
from dataclasses import dataclass, is_dataclass, replace
from .utils import IterableWrapper, StepResult

logger = get_logger(__file__)

T = TypeVar("T")
S = TypeVar("S", bound=Space)
# TODO: Add 'TensorSpace' space which wraps a given space, doing the same kinda thing
# as in Sparse.


class ConvertToFromTensors(IterableWrapper):
    """ Wrapper that converts Tensors into samples/ndarrays and vice versa.

    Whatever comes into the env is converted into np.ndarrays or samples from
    the action space, and whatever comes out of the environment (observations,
    rewards, dones, etc.) get converted to Tensors.

    Also supports Dict/Tuple/etc observation/action spaces.

    Also makes it so the `sample` methods of both the observation and
    action spaces return Tensors, and that their `contains` methods also accept
    Tensors as an input.

    If `device` is given, created Tensors are moved to the provided device.
    """

    def __init__(self, env: gym.Env, device: Union[torch.device, str] = None):
        super().__init__(env=env)
        self.device = device
        self.observation_space: Space = add_tensor_support(
            self.env.observation_space, device=device
        )
        self.action_space: Space = add_tensor_support(
            self.env.action_space, device=device
        )
        self.reward_space: Space
        if hasattr(self.env, "reward_space"):
            self.reward_space = self.env.reward_space
        else:
            reward_range = getattr(self.env, "reward_range", (-np.inf, np.inf))
            reward_shape: Tuple[int, ...] = ()
            if self.is_vectorized:
                reward_shape = (self.env.num_envs,)
            self.reward_space = spaces.Box(
                reward_range[0], reward_range[1], reward_shape, np.float32
            )
        self.reward_space = add_tensor_support(self.reward_space, device=device)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self.observation(obs)

    def observation(self, observation):
        return to_tensor(self.observation_space, observation, device=self.device)

    def action(self, action):
        if isinstance(self.action_space, spaces.MultiDiscrete) and is_dataclass(action):
            # TODO: Fixme, the actions don't currently fit their space!
            action_np = replace(action, y_pred=from_tensor(self.action_space, action.y_pred))
            # FIXME: for now, unwrapping the actions
            action = action_np["y_pred"]
            return action
        return from_tensor(self.action_space, action)

    def reward(self, reward):
        # FIXME: This doesn't exactly work when our 'reward space' isn't a dict and
        # 'reward' is a Batch object, and might also be the same with the actions above
        if isinstance(self.reward_space, spaces.MultiDiscrete) and is_dataclass(reward):
            return replace(
                reward, y=to_tensor(self.reward_space, reward.y, device=self.device)
            )
        return to_tensor(self.reward_space, reward, device=self.device)

    def step(self, action: Tensor) -> StepResult:
        action = self.action(action)
        assert action in self.env.action_space, (action, self.env.action_space)

        result = self.env.step(action)
        observation, reward, done, info = result
        observation = self.observation(observation)
        reward = self.reward(reward)
        # NOTE: Not sure this is useful, actually!
        # done = torch.as_tensor(done, device=self.device)
        
        # We could actually do this!
        # info = np.ndarray(info)
        if isinstance(result, StepResult):
            return type(result)(
                **dict(observation=observation, reward=reward, done=done, info=info)
            )
        else:
            return StepResult(observation, reward, done, info)


def supports_tensors(space: S) -> bool:
    # TODO: Remove this, instead use a generic function
    return getattr(space, "_supports_tensors", False)


def has_tensor_support(space: S) -> bool:
    return supports_tensors(space)


def _mark_supports_tensors(space: S) -> None:
    # TODO: Remove this!
    setattr(space, "_supports_tensors", True)


@singledispatch
def add_tensor_support(space: S, device: torch.device = None) -> S:
    """Modifies `space` so its `sample()` method produces Tensors, and its
    `contains` method also accepts Tensors.

    For Dict and Tuple spaces, all the subspaces are also modified recursively.

    Returns the modified Space.
    """
    # Save the original methods so we can use them.
    sample = space.sample
    contains = space.contains
    if supports_tensors(space):
        # logger.debug(f"Space {space} already supports Tensors.")
        return space

    @wraps(space.sample)
    def _sample(*args, **kwargs):
        samples = sample(*args, **kwargs)
        samples = to_tensor(space, samples)
        if device:
            samples = move(samples, device)
        return samples

    @wraps(space.contains)
    def _contains(x: Union[Tensor, Any]) -> bool:
        x = from_tensor(space, x)
        return contains(x)

    space.sample = _sample
    space.contains = _contains
    _mark_supports_tensors(space)
    assert has_tensor_support(space)
    return space


@add_tensor_support.register
def _(space: Image, device: torch.device = None) -> Image:
    tensor_box = TensorBox(
        space.low, space.high, shape=space.shape, dtype=space.dtype, device=device
    )
    return ImageTensorSpace.from_box(tensor_box)


@add_tensor_support.register
def _(space: spaces.Dict, device: torch.device = None) -> spaces.Dict:
    space = type(space)(
        **{
            key: add_tensor_support(value, device=device)
            for key, value in space.spaces.items()
        }
    )
    # TODO: Remove this '_mark_supports_tensors' and instead use a generic function.
    _mark_supports_tensors(space)
    return space


@add_tensor_support.register
def _(space: TypedDictSpace, device: torch.device = None) -> TypedDictSpace:
    space = type(space)(
        {
            key: add_tensor_support(value, device=device)
            for key, value in space.spaces.items()
        },
        dtype=space.dtype,
    )
    _mark_supports_tensors(space)
    return space


@add_tensor_support.register(NamedTupleSpace)
def _(space: Dict, device: torch.device = None) -> Dict:
    space = type(space)(
        **{
            key: add_tensor_support(value, device=device)
            for key, value in space.items()
        },
        dtype=space.dtype,
    )
    _mark_supports_tensors(space)
    return space


@add_tensor_support.register(spaces.Tuple)
def _(space: Dict, device: torch.device = None) -> Dict:
    space = type(space)(
        [add_tensor_support(value, device=device) for value in space.spaces]
    )
    _mark_supports_tensors(space)
    return space

# TODO: Should this be moved to the place where these are defined instead?
from sequoia.common.spaces.tensor_spaces import (
    TensorBox,
    TensorDiscrete,
    TensorMultiDiscrete,
)


@add_tensor_support.register
def _(space: spaces.Box, device: torch.device = None) -> spaces.Box:
    space = TensorBox(
        space.low, space.high, shape=space.shape, dtype=space.dtype, device=device
    )
    _mark_supports_tensors(space)
    return space


@add_tensor_support.register
def _(space: spaces.Discrete, device: torch.device = None) -> spaces.Box:
    space = TensorDiscrete(n=space.n, device=device)
    _mark_supports_tensors(space)
    return space


@add_tensor_support.register
def _(space: spaces.MultiDiscrete, device: torch.device = None) -> spaces.Box:
    space = TensorMultiDiscrete(nvec=space.nvec, device=device)
    _mark_supports_tensors(space)
    return space
