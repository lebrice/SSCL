from pathlib import Path
from typing import Callable, List, Optional, Tuple

import gym
import numpy as np
import pytest
from gym import spaces

from sequoia.common.config import Config
from sequoia.common.spaces import Sparse
from sequoia.common.transforms import ChannelsFirstIfNeeded, ToTensor, Transforms
from sequoia.conftest import xfail_param, param_requires_atari_py
from sequoia.utils.utils import take
from sequoia.common.gym_wrappers.convert_tensors import has_tensor_support

from .continual_rl_setting import ContinualRLSetting



def test_task_schedule_is_used():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = ContinualRLSetting(dataset="CartPole-v0", max_steps = 100, steps_per_task=10, nb_tasks=10)
    env = setting.train_dataloader(batch_size=None)

    
    starting_length = env.length
    assert starting_length == 0.5
    
    observations = env.reset()
    lengths: List[float] = []
    for i in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
        length = env.length
        lengths.append(length)
    assert not all(length == starting_length for length in lengths)
    

@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "dataset, expected_obs_shape", [
        ("CartPole-v0", (3, 400, 600)),
        # param_requires_atari_py("Breakout-v0", (3, 210, 160)), 
        param_requires_atari_py("Breakout-v0", (3, 84, 84)), # Since the AtariWrapper gets added by default
        # ("duckietown", (120, 160, 3)),
    ],
)
def test_check_iterate_and_step(dataset: str,
                                expected_obs_shape: Tuple[int, ...],
                                batch_size: int):
    """ Test that the observations are of the right type and shape, regardless
    of wether we iterate on the env by calling 'step' or by using it as a
    DataLoader.
    """
    setting = ContinualRLSetting(dataset=dataset)
    
    expected_obs_batch_shape = (batch_size, *expected_obs_shape)
    if batch_size is None:
        expected_obs_batch_shape = expected_obs_shape
    
    # Test the shapes of the obs generated by the train/val/test dataloaders.
    dataloader_methods = [
        setting.train_dataloader,
        setting.val_dataloader,
        setting.test_dataloader
    ]
    assert setting.nb_tasks == 1
    
    with setting.train_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)
        obs = temp_env.reset()
        # BUG:
        # assert has_tensor_support(temp_env.observation_space)
        assert obs[0].shape == temp_env.observation_space[0].shape
        

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)

    # NOTE: Limitting the batch size at test time to None (i.e. a single env)
    # because of how the Monitor class works atm.
    with setting.test_dataloader(batch_size=None) as temp_env:
        assert temp_env.observation_space[0] == spaces.Box(0., 1., expected_obs_shape, dtype=np.float32)
        assert type(temp_env.observation_space)
        # assert temp_env.observation_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)

    def check_obs(obs):
        assert isinstance(obs, ContinualRLSetting.Observations), obs[0].shape
        assert obs.x.shape == expected_obs_batch_shape
        assert obs.task_labels is None or all(task_label == None for task_label in obs.task_labels)
    
    # FIXME: Same a temp copy
    expected_obs_batch_shape_ = expected_obs_batch_shape
    
    for dataloader_method in dataloader_methods:
        print(f"Testing dataloader method {dataloader_method.__name__}")
        ## FIXME: Remove this if we allow batched env at test time. 
        if dataloader_method.__name__ == "test_dataloader":
            # Temporarily change the expected shape.
            expected_obs_batch_shape = expected_obs_shape
            env = dataloader_method(batch_size=None)
            assert env.batch_size is None
            
        else:
            # Restore the original value.
            expected_obs_batch_shape = expected_obs_batch_shape_
            env = dataloader_method(batch_size=batch_size)
            assert env.batch_size == batch_size
        ##
        # env = dataloader_method(batch_size=batch_size)

        reset_obs = env.reset()
        check_obs(reset_obs)

        step_obs, *_ = env.step(env.action_space.sample())
        check_obs(step_obs)

        for iter_obs in take(env, 3):
            check_obs(iter_obs)
            reward = env.send(env.action_space.sample())


@pytest.mark.xfail(reason=f"TODO: DQN model only accepts string environment names...")
def test_dqn_on_env(tmp_path: Path):
    """ TODO: Would be nice if we could have the models work directly on the
    gym envs..
    """
    from pl_bolts.models.rl import DQN
    from pytorch_lightning import Trainer
    setting = ContinualRLSetting(observe_state_directly=False)
    env = setting.train_dataloader(batch_size=None)
    model = DQN(env)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmp_path)
    success = trainer.fit(model)
    assert success == 1


def test_passing_task_schedule_sets_other_attributes_correctly():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = ContinualRLSetting(dataset="CartPole-v0", train_task_schedule={
        0: {"gravity": 5.0},
        100: {"gravity": 10.0},
        200: {"gravity": 20.0},
    })
    assert setting.phases == 1
    assert setting.nb_tasks == 2
    assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        5_000: {"gravity": 10.0},
        10_000: {"gravity": 20.0},
    }
    assert setting.test_steps == 10_000
    assert setting.test_steps_per_task == 5_000


from sequoia.settings.assumptions.incremental_test import DummyMethod
from sequoia.conftest import DummyEnvironment


def test_fit_and_on_task_switch_calls():
    setting = ContinualRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        steps_per_task=100,
        max_steps=500,
        test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = DummyMethod()
    results = setting.apply(method)
    # == 30 task switches in total.
    assert method.n_task_switches == 0
    assert method.n_fit_calls == 1 # TODO: Add something like this.
    assert not method.received_task_ids 
    assert not method.received_while_training
