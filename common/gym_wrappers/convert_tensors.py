from functools import singledispatch, wraps
from typing import Any, Dict, List, Tuple, TypeVar, Union, Optional

import gym
import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor
from utils.generic_functions import move
from utils.logging_utils import get_logger

logger = get_logger(__file__)

S = TypeVar("S", bound=Space)

from utils.generic_functions import to_tensor, from_tensor


class ConvertToFromTensors(gym.Wrapper):
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
        self.observation_space: Space = add_tensor_support(self.env.observation_space, device=device)
        self.action_space: Space = add_tensor_support(self.env.action_space, device=device)
        if hasattr(self.env, "reward_space"):
            self.reward_space: Space = add_tensor_support(self.env.reward_space, device=device)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return to_tensor(self.observation_space, obs, device=self.device)

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict]]:
        action = from_tensor(self.action_space, action)
        assert action in self.env.action_space, (action, self.env.action_space)
        
        result = self.env.step(action)
        observation, reward, done, info = result
        
        observation = to_tensor(self.observation_space, observation, self.device)

        if hasattr(self, "reward_space"):
            reward = to_tensor(self.reward_space, reward, self.device)
        else:
            reward = torch.as_tensor(reward, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        # We could actually do this!
        # info = np.ndarray(info)
        return type(result)([observation, reward, done, info])


def supports_tensors(space: S) -> bool:
    return getattr(space, "__supports_tensors", False)


def _mark_supports_tensors(space: S) -> bool:
    return setattr(space, "__supports_tensors", True)


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
        logger.debug(f"Space {space} already supports Tensors.")
        return space
    _mark_supports_tensors(space)
    
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
    if isinstance(space, (spaces.Tuple, spaces.Dict)):
        # Also add tensor support to all the subspaces.
        @wraps(space.__getitem__)
        def __getitem__(self, index):
            return add_tensor_support(self.spaces[index])
        space.__getitem__ = __getitem__
    
    return space
