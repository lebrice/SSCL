""" Typed protocols versions of the classes in Gym. All gym classes match these protocols.

NOTE: These are only used for type-hints, and  will be True
"""
from typing import (
    List,
    Optional,
    TypeVar,
    runtime_checkable,
)
from typing import Any, Protocol, Sequence, Tuple
import numpy as np


Observation = TypeVar("Observation")
_Observation_co = TypeVar("Observation_co", covariant=True)
_Action = TypeVar("Action")
_Reward = TypeVar("Reward")
Reward_co = TypeVar("Reward_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


def detach(value: T) -> T:
    if hasattr(value, "detach") and callable(value.detach):  # type: ignore
        return value.detach()  # type: ignore
    return value


@runtime_checkable
class _Space(Protocol[T_co]):
    def sample(self) -> T_co:
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        raise NotImplementedError
    
    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def seed(self, seed: Optional[int]) -> List[int]:
        raise NotImplementedError


@runtime_checkable
class _Env(Protocol[Observation, _Action, Reward_co]):
    observation_space: _Space[Observation]
    action_space: _Space[_Action]

    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action: _Action) -> Tuple[Observation, Reward_co, bool, dict]:
        raise NotImplementedError

    def seed(self, seed: Optional[int]) -> List[int]:
        seeds = []
        seeds.extend(self.action_space.seed(seed))
        seeds.extend(self.observation_space.seed(seed))
        return seeds
    
    @property
    def unwrapped(self) -> "Env":
        return self

@runtime_checkable
class _VectorEnv(_Env[Observation, _Action, Reward_co], Protocol):
    num_envs: int

    def step(  # type: ignore
        self, action: _Action
    ) -> Tuple[Observation, Reward_co, np.ndarray, Sequence[dict]]:
        pass

# import gym
# VectorEnv.register(gym.vector.VectorEnv)
# Env.register(gym.Env)