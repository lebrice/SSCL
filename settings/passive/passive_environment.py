"""TODO: Creates a Gym Environment (and DataLoader) from a traditional
Supervised dataset. 
"""

from typing import *

import gym
import torch
import numpy as np
from gym import spaces
from torch.utils.data import DataLoader, Dataset, IterableDataset

from common.transforms import Compose
from utils.logging_utils import get_logger

from ..base.environment import (Actions, ActionType, Environment, Observations,
                                ObservationType, Rewards, RewardType)
from torch.utils.data.dataloader import _BaseDataLoaderIter
logger = get_logger(__file__)


class PassiveEnvironment(DataLoader, gym.Env, Environment[Tuple[ObservationType,
                                                          Optional[ActionType]],
                                                 ActionType,
                                                 RewardType]):
    """Environment in which actions have no influence on future observations.

    Can either be iterated on like a normal DataLoader, in which case it gives
    back the observation and the reward at the same time, or as a gym
    Environment, in which case it gives the rewards and the next batch of
    observations once an action is given.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this
    category. Similarly to Environment, this just adds some methods on top of
    the usual PyTorch DataLoader.
    """
    passive: ClassVar[bool] = True
    
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self,
                 dataset: Union[IterableDataset, Dataset],
                 split_batch_fn: Callable[[Tuple[Any, ...]], Tuple[ObservationType, ActionType]] = None,
                 observation_space: gym.Space = None,
                 action_space: gym.Space = None,
                 reward_space: gym.Space = None,
                 n_classes: int = None,
                #  pretend_to_be_active: bool = False,
                 **kwargs):
        """Creates the DataLoader/Environment for the given dataset.

        Parameters
        ----------
        dataset : [type]
            The dataset to iterate on. Should ideally be indexable (a Map-style
            dataset).

        TODO: IDK if this is the best way to go about it. Maybe it would be best
        to just adopt the gym Env API and forget about this "send" idea here. 
        pretend_to_be_active : bool, optional
            Wether to withhold the rewards (labels) from the batches when being
            iterated on like the usual dataloader, and to only give them back
            after an action is received through the 'send' method. False by
            default, in which case this behaves exactly as a normal dataloader
            when being iterated on.

        **kwargs:
            The rest of the usual DataLoader kwargs.
        """
        self.split_batch_fn = split_batch_fn
        
        # TODO: When True, withold the labels from the yielded batches until a
        # prediction is received through in the 'send' method.
        # self.pretend_to_be_active = pretend_to_be_active
        # self.observations_type = observations_type
        # self.actions_type = actions_type
        # self.rewards_type = rewards_type

        # self.batch_transforms: List[Callable] = batch_transforms or []
        # from common.transforms import SplitBatch
        # if not any(isinstance(t, SplitBatch) for t in self.batch_transforms):
            # if observations_type and rewards_type:
            #     self.batch_transforms.append(SplitBatch(observations_type, rewards_type))
            # else:
            #     raise RuntimeError(
            #         f"`batch_transforms` needs to contain a SplitBatch "
            #         f"transform! Or, you can pass an `observations_type` and a"
            #         f"`rewards_type` to the {__class__} constructor and it "
            #         f"will create one for you. \n"
            #         f"(transforms: {self.batch_transforms})"
            #     )
        # self.batch_transforms = Compose(self.batch_transforms)

        super().__init__(dataset=dataset, **kwargs)
        # self.observations: Union[Observations, Any] = None
        # self.rewards: Union[Rewards, Any] = None

        if not all([observation_space, action_space, reward_space]):
            if isinstance(self.dataset, IterableDataset):
                temp_iterator = iter(self.dataset)
                sample = next(temp_iterator)
                del temp_iterator
            else:
                sample = self.dataset[0]

            if not isinstance(sample, (tuple, list)) or len(sample) == 1:
                # IDEA: In this case I guess we could require the action/reward
                # spaces to be passed through the constructor arguments?
                raise NotImplementedError(
                    "Can't use the PassiveEnvironment DataLoader/gym.Env hybrid "
                    "on unsupervised datasets yet. "
                )
            if len(sample) > 2:
                if split_batch_fn:
                    sample_batch = split_batch_fn(sample_batch)
                else:
                    raise RuntimeError(
                        "You need to give a `split_batch_fn` to be used whenever "
                        "there are more than 2 tensors per batch."
                    )
            observation, rewards = sample

            if observation_space is None:
                assert observation.min() >= 0. # Assuming this for now.
                assert observation.max() <= 1.
                observation_space = spaces.Box(low=0, high=1, shape=observation.shape)

            if not action_space and not reward_space:
                # TODO: Need to know how many classes there are in the dataset!
                if n_classes is None:
                    raise RuntimeError(
                        "Need to have n_classes passed when neither of "
                        "action_space or reward_space are given."
                    )
                # TODO: This also assumes that the rewards above is an int or an int tensor.
                # assert False, rewards
                if isinstance(rewards, int) or not torch.is_floating_point(rewards):
                    action_space = reward_space = spaces.Discrete(n=n_classes)
                else:
                    assert False, rewards
            elif not action_space:
                action_space = reward_space
            else:
                reward_space = action_space

        # Batch these spaces to reflect the batch size.
        # TODO: Should we be doing this? This is so we match the AsyncVectorEnv
        # from the gym.vector API.
        # NOTE: On the last batch, if drop_last = False, the observations/actions
        # will not reflect the spaces. Not sure if this could be a problem later.    
        self.observation_space = spaces.Tuple([
            observation_space for _ in range(self.batch_size)
        ])
        self.action_space = spaces.Tuple([
            action_space for _ in range(self.batch_size)
        ])
        self.reward_space = spaces.Tuple([
            reward_space for _ in range(self.batch_size)
        ])

        self._iterator: Optional[_BaseDataLoaderIter] = None
        self._previous_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._current_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._next_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._done: bool = False
        self._closed: bool = False

    def __getitem__(self, index):
        return self.dataset[index]

    def reset(self) -> ObservationType:
        """ Resets the env by deleting and re-creating the iterator. """
        if self._iterator:
            self._iterator.close()
        del self._iterator
        self._iterator = self.__iter__()

        self._previous_batch = None
        self._current_batch = next(self._iterator)
        return self._current_batch[0]

    def close(self) -> None:
        self._closed = True

    def __next__(self) -> Tuple[ObservationType, RewardType]:
        """Gets the next batch from the underlying dataset.

        Returns the result of the `split_batch_fn`, if set.

        Returns
        -------
        Tuple[ObservationType, RewardType]
            [description]
        """
        if self._iterator is None:
            self._iterator = self.__iter__()
        batch = next(self._iterator)
        if self.split_batch_fn:
            return self.split_batch_fn(batch)
        return batch
    
    def step(self, action: ActionType) -> Tuple[ObservationType, RewardType, bool, Dict]:
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Env is closed!")
        if self._done:
            raise gym.error.ResetNeeded("Need to reset the env, as it is done!")

        # IDEA: Let subclasses customize how the action impacts the env?
        self.use_action(action)

        # NOTE: This prev/current/next setup is so we can give the right 'done'
        # signal.
        self._previous_batch = self._current_batch
        if self._next_batch is None:
            # This should only ever happen right after resetting.
            self._next_batch = next(self, None)
        self._current_batch = self._next_batch
        self._next_batch = next(self, None)

        # TODO: Return done=True when the iterator is exhausted?
        self._done = self._next_batch is None
        obs = self._current_batch[0]
        reward = self._previous_batch[1]
        # Empty for now I guess?
        info = {}
        return obs, reward, self._done, info

    def use_action(self, action):
        """ Override this method if you want the actions to affect the env
        somehow. (You could also just override the `step` method, I guess).
        
        Parameters
        ----------
        action : [type]
            [description]
        """

    def get_info(self) -> Dict:
        """Returns the dict to be returned as the 'info' in step().

        IDEA: We could subclass this to change whats in the 'info' dict, maybe
        add some task information?
        
        Returns
        -------
        Dict
            [description]
        """
        return {}

    def render(self, mode="rgb_array") -> np.ndarray:
        return self._current_batch.cpu().numpy()        
    
    def __iter__(self) -> Iterable[Tuple[ObservationType, Optional[RewardType]]]:
        """Iterate over the dataset, yielding batches of Observations and
        Rewards, just like a regular DataLoader.
        """
        return super().__iter__()
        # for batch in super().__iter__():
        #     batch = self.batch_transforms(batch)
            
        #     # For now, just to simplify, we assume that the batch has already
        #     # been split into Observations and Actions by a SplitBatch transform.
        #     assert len(batch) == 2
        #     assert isinstance(batch[0], Observations)
        #     assert isinstance(batch[1], Rewards)
        #     self.observations, self.rewards = batch
            
        #     if self.pretend_to_be_active:
        #         # TODO: Should we yield one item, or two?
        #         yield self.observations, None
        #     else:
        #         yield self.observations, self.rewards

    def send(self, action: Actions) -> Rewards:
        """ Return the last latch of rewards from the dataset (which were
        withheld if in 'active' mode)
        """
        assert False, (
            "TODO: work in progress, if you're gonna pretend this is an "
            "'active' environment, then use the gym API for now."
        )
        # assert self.action_space.contains(action), action
        return None

    def close(self):
        pass
