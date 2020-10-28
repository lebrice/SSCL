from typing import Callable, List, Optional, Tuple

import gym
import pytest
import numpy as np
from gym import spaces

from common.gym_wrappers.sparse_space import Sparse
from common.config import Config
from common.transforms import ChannelsFirstIfNeeded, ToTensor, Transforms
from conftest import xfail_param
from utils.utils import take

from .class_incremental_rl_setting import ClassIncrementalRLSetting


@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "dataset, expected_obs_shape", [
        ("CartPole-v0", (3, 400, 600)),
        ("Breakout-v0", (3, 210, 160)),
        # ("duckietown", (120, 160, 3)),
    ],
)
def test_check_iterate_and_step(dataset: str,
                                expected_obs_shape: Tuple[int, ...],
                                batch_size: int):
    setting = ClassIncrementalRLSetting(dataset=dataset, nb_tasks=5)
    assert len(setting.train_task_schedule) == 5
    assert not setting.smooth_task_boundaries
    assert setting.task_labels_at_train_time
    
    # TODO: Should we have the task label space in this case?
    assert setting.task_labels_at_train_time
    assert not setting.task_labels_at_test_time
    
    if batch_size is None:
        expected_obs_batch_shape = expected_obs_shape
    else:
        expected_obs_batch_shape = (batch_size, *expected_obs_shape)
    
    with setting.train_dataloader(batch_size=batch_size) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)
        assert obs_space[1] == spaces.MultiDiscrete([5] * batch_size) if batch_size else spaces.Discrete(5)

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        # No task labels:
        obs_space = temp_env.observation_space

        assert obs_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)
        if batch_size:
            assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), none_prob=1.) for _ in range(batch_size)]))
        else:
            assert obs_space[1] == Sparse(spaces.Discrete(5), none_prob=1.)

    with setting.test_dataloader(batch_size=batch_size) as temp_env:
        obs_space = temp_env.observation_space
        
        # No task labels:
        if batch_size:
            assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), none_prob=1.) for _ in range(batch_size)]))
        else:
            assert obs_space[1] == Sparse(spaces.Discrete(5), none_prob=1.)

    # with setting.val_dataloader(batch_size=batch_size) as valid_env:
    #     assert str(valid_env.observation_space) == str(spaces.Tuple([
    #         spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32),
    #         Sparse(spaces.MultiDiscrete([5] * batch_size), none_prob=1.),
    #     ]))

    def check_obs(obs, task_label: int = None):
        assert isinstance(obs, ClassIncrementalRLSetting.Observations), obs[0].shape
        assert obs.x.shape == expected_obs_batch_shape
        if batch_size is None:
            assert obs.task_labels is task_label
        else:
            assert obs.task_labels is task_label or all(task_label == task_label for task_label in obs.task_labels)

    env = setting.train_dataloader(batch_size=batch_size)
    reset_obs = env.reset()
    check_obs(reset_obs, task_label=0)
    
    for i in range(5):
        step_obs, *_ = env.step(env.action_space.sample())
        check_obs(step_obs, task_label=0)

    for iter_obs in take(env, 3):
        check_obs(iter_obs, task_label=0)
        reward = env.send(env.action_space.sample())
        env.render("human")
        
    env.close()


from settings import Method
from .continual_rl_setting import TestEnvironment


class DummyMethod(Method, target_setting=ClassIncrementalRLSetting):
    def __init__(self):
        self.n_task_switches = 0
        self.received_task_ids: List[Optional[int]] = []
    
    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        obs = train_env.reset()
        for i in range(100):
            obs, reward, done, info = train_env.step(train_env.action_space.sample())
            if done:
                break
    
    def test(self, test_env: TestEnvironment):
        
        while not test_env.is_closed():
            done = False
            obs = test_env.reset()
            while not done:
                actions = test_env.action_space.sample()
                obs, _, done, info = test_env.step(actions)

    def get_actions(self, observations: ClassIncrementalRLSetting.Observations, action_space: gym.Space):
        return action_space.sample()

    def on_task_switch(self, task_id: int=None):
        self.n_task_switches += 1
        self.received_task_ids.append(task_id)

from conftest import DummyEnvironment
def test_on_task_switch_is_called():
    dataset = "Breakout-v0"
    setting = ClassIncrementalRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        n_steps_per_task=10,
        max_steps=50,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = DummyMethod()
    results = setting.apply(method)
    
    assert method.n_task_switches == 5
    assert method.received_task_ids == list(range(5))
    
