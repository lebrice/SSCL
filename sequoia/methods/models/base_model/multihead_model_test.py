"""Tests for the class-incremental version of the Model class.
"""
# from sequoia.conftest import config
from collections import defaultdict
from typing import Dict, List, Tuple, Type, Optional

import gym
import numpy as np
import pytest
import torch
from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import TaskSet
from gym import spaces
from gym.spaces.utils import flatdim
from gym.vector import SyncVectorEnv
from gym.wrappers import TimeLimit
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from sequoia.common import Loss
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import MultiTaskEnvironment
from sequoia.methods.base_method import BaseMethod
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.methods.models.output_heads.rl.episodic_a2c import EpisodicA2C
from sequoia.settings import ClassIncrementalSetting, TraditionalRLSetting, RLSetting
from sequoia.settings.base import Environment
from sequoia.settings.rl import IncrementalRLSetting
from sequoia.utils import take

from .base_model import BaseModel
from .multihead_model import MultiHeadModel, OutputHead, get_task_indices


@pytest.fixture()
def mixed_samples(config: Config):
    """ Fixture that produces some samples from each task. """
    dataset = MNIST(config.data_dir, download=True, train=True)
    datasets: List[TaskSet] = ClassIncremental(dataset, nb_tasks=5)
    n_samples_per_task = 10
    indices = list(range(10))
    samples_per_task: Dict[int, Tensor] = {
        i: tuple(map(torch.as_tensor, taskset.get_samples(indices)))
        for i, taskset in enumerate(datasets)
    }
    return samples_per_task


class MockOutputHead(OutputHead):
    def __init__(self, *args, Actions: Type, task_id: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.Actions = Actions
        self.name = f"task_{task_id}"

    def forward(self, observations, representations) -> Tensor:  # type: ignore
        """ This mock forward just creates an action that is related to the observation
        and the task id for this output head.
        """
        x: Tensor = observations.x
        assert (observations.task_labels == self.task_id).all()
        h_x = representations
        # actions = torch.stack([h_i.mean() * self.task_id for h_i in h_z])
        # actions = torch.stack([x_i.mean() * self.task_id for x_i in x])
        actions = [x_i.mean() * self.task_id for x_i in x]
        actions = torch.stack(actions)
        fake_logits = torch.rand([actions.shape[0], self.action_space.n])
        from sequoia.methods.models.output_heads.classification_head import (
            ClassificationOutput,
        )
        # assert issubclass(ClassificationOutput, self.Actions)
        # TODO: Ideally self.Actions would already be a subclass of ClassificationActions!
        # return self.Actions(y_pred=actions, logits=fake_logits)
        return ClassificationOutput(y_pred=actions, logits=fake_logits)

    def get_loss(self, forward_pass, actions, rewards):
        return Loss(self.name, 0.0)


# def mock_output_task(self: MultiHeadModel, x: Tensor, h_x: Tensor) -> Tensor:
#     return self.output_head(x)

# def mock_encoder(self: MultiHeadModel, x: Tensor) -> Tensor:
#     return x.new_ones(self.hp.hidden_size)


@pytest.mark.parametrize(
    "indices",
    [
        slice(0, 10),  # all the same task (0)
        slice(0, 20),  # 10 from task 0, 10 from task 1
        slice(0, 30),  # 10 from task 0, 10 from task 1, 10 from task 2
        slice(0, 50),  # 10 from each task.
    ],
)
def test_multiple_tasks_within_same_batch(
    mixed_samples: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    indices: slice,
    monkeypatch,
    config: Config,
):
    """ TODO: Write out a test that checks that when given a batch with data
    from different tasks, and when the model is multiheaded, it will use the
    right output head for each image.
    """
    # Get a mixed batch
    xs, ys, ts = map(torch.cat, zip(*mixed_samples.values()))
    xs = xs[indices]
    ys = ys[indices]
    ts = ts[indices].int()
    obs = ClassIncrementalSetting.Observations(x=xs, task_labels=ts)

    setting = ClassIncrementalSetting()
    model = MultiHeadModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(batch_size=30, multihead=True),
        config=config,
    )

    class MockEncoder(nn.Module):
        def forward(self, x: Tensor):
            return x.new_ones([x.shape[0], model.hidden_size])

    mock_encoder = MockEncoder()
    model.encoder = mock_encoder

    for i in range(5):
        model.output_heads[str(i)] = MockOutputHead(
            input_space=spaces.Box(0, 1, [model.hidden_size]),
            action_space=spaces.Discrete(2),
            Actions=setting.Actions,
            task_id=i,
        )
    model.output_head = model.output_heads["0"]

    forward_pass = model(obs)
    y_preds = forward_pass["y_pred"]

    assert y_preds.shape == ts.shape
    assert torch.all(y_preds == ts * xs.view([xs.shape[0], -1]).mean(1))


def test_multitask_rl_bug_without_PL(monkeypatch):
    """ TODO: on_task_switch is called on the new observation, but we need to produce a
    loss for the output head that we were just using!
    """
    # NOTE: Tasks don't have anything to do with the task schedule. They are sampled at
    # each episode.
    max_episode_steps = 5
    setting = TraditionalRLSetting(
        dataset="cartpole",
        batch_size=1,
        nb_tasks=2,
        train_max_steps=100,
        max_episode_steps=max_episode_steps,
        add_done_to_observations=True,
    )
    assert setting.stationary_context

    # setting = RLSetting.load_benchmark("monsterkong")
    config = Config(debug=True, verbose=True, seed=123)
    config.seed_everything()
    model = BaseModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(
            multihead=True,
            output_head=EpisodicA2C.HParams(accumulate_losses_before_backward=True),
        ),
        config=config,
    )
    # TODO: Maybe add some kind of "hook" to check which losses get returned when?
    model.train()
    # from pytorch_lightning import Trainer
    # trainer = Trainer(fast_dev_run=True)
    # trainer.fit(model, train_dataloader=setting.train_dataloader())
    # trainer.setup(model, stage="fit")

    # from pytorch_lightning import Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    episodes = 0
    max_episodes = 5

    # Dict mapping from step to loss at that step.
    losses: Dict[int, Loss] = {}

    with setting.train_dataloader() as env:
        env.seed(123)
        # env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Iterate over the environment, which yields one observation at a time:
        for step, obs in enumerate(env):
            assert isinstance(obs, RLSetting.Observations)

            if step == 0:
                assert not any(obs.done)
            start_task_label = obs["task_labels"][0]

            stored_steps_in_each_head_before = {
                task_key: output_head.num_stored_steps(0)
                for task_key, output_head in model.output_heads.items()
            }
            forward_pass: ForwardPass = model.forward(observations=obs)
            rewards = env.send(forward_pass.actions)

            loss: Loss = model.get_loss(
                forward_pass=forward_pass, rewards=rewards, loss_name="debug"
            )
            stored_steps_in_each_head_after = {
                task_key: output_head.num_stored_steps(0)
                for task_key, output_head in model.output_heads.items()
            }
            # if step == 5:
            #     assert False, (loss, stored_steps_in_each_head_before, stored_steps_in_each_head_after)

            if any(obs.done):
                assert loss.loss != 0.0, step
                assert loss.loss.requires_grad

                # Backpropagate the loss, update the models, etc etc.
                loss.loss.backward()
                model.on_after_backward()
                optimizer.step()
                model.on_before_zero_grad(optimizer)
                optimizer.zero_grad()

                # TODO: Need to let the model know than an update is happening so it can clear
                # buffers etc.

                episodes += sum(obs.done)
                losses[step] = loss
            else:
                assert loss.loss == 0.0
            # TODO:
            print(
                f"Step {step}, episode {episodes}: x={obs.x}, done={obs.done}, reward={rewards} task labels: {obs.task_labels}, loss: {loss.losses.keys()}: {loss.loss}"
            )

            if episodes > max_episodes:
                break
    # assert False, losses

# @pytest.mark.xfail(
#     reason=f"Not quite sure why, but getting weird device bug here *sometimes*:"
# )
def test_multitask_rl_bug_with_PL(monkeypatch, config: Config):
    """
    """
    # NOTE: Tasks don't have anything to do with the task schedule. They are sampled at
    # each episode.

    cpu_config = config
    # cpu_config = Config(device="cpu", num_workers=0)

    setting = TraditionalRLSetting(
        dataset="cartpole",
        batch_size=1,
        num_workers=0,
        nb_tasks=2,
        train_max_steps=200,
        test_max_steps=200,
        max_episode_steps=5,
        add_done_to_observations=True,
        config=cpu_config,
    )
    assert setting.train_max_steps == 200
    assert setting.test_max_steps == 200
    assert setting.stationary_context

    # setting = RLSetting.load_benchmark("monsterkong")
    cpu_config.seed_everything()
    model = BaseModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(
            multihead=True,
            output_head=EpisodicA2C.HParams(accumulate_losses_before_backward=True),
        ),
        config=cpu_config,
    ).to(device=config.device)

    # TODO: Maybe add some kind of "hook" to check which losses get returned when?
    model.train()
    assert not model.automatic_optimization

    from pytorch_lightning import Trainer

    # Import this and use it to create the Trainer, rather than creating the Trainer
    # directly, so we don't get the same bug (due to with_is_last in PL) from the
    # DataConnector.
    from sequoia.methods.base_method import TrainerConfig

    # NOTE: We only do this so that the Model has a self.trainer attribute and so the
    # model.training_step below can be used:
    if config.device.type == "cuda":
        trainer_config = TrainerConfig(fast_dev_run=True)
    else:
        trainer_config = TrainerConfig(
            fast_dev_run=True, gpus=0, distributed_backend=None,
        )

    trainer = trainer_config.make_trainer(config=cpu_config)

    # Fit in 'fast_dev_run' mode, so just a single batch of train / valid / test data.
    with setting.train_dataloader() as temp_env:
        temp_env.seed(123)
        trainer.fit(model, train_dataloader=temp_env)

    #NOTE: If we don't clear the buffers, there is a bug because the things that get put
    # in buffers aren't on the same device as later.
    model.output_head.clear_all_buffers()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    episodes = 0
    max_episodes = 5

    # Dict mapping from step to loss at that step.
    losses: Dict[int, List[Loss]] = defaultdict(list)

    with setting.train_dataloader() as env:
        env.seed(123)

        # TODO: Interesting bug/problem: Since the VectorEnvs always want to reset the
        # env at the end of the episode, they also also so on the individual envs.
        # In order to solve that, we need to NOT put any 'ActionLimit' on the inside
        # envs, but only on the outer env.
        for step, obs in enumerate(env):
            assert isinstance(obs, RLSetting.Observations)
            from sequoia.utils.generic_functions import to_tensor

            print(step, env.is_closed())
            forward_pass = model.training_step(batch=obs, batch_idx=step)
            step_results: Optional[Loss] = model.training_step_end([forward_pass])
            loss_tensor: Optional[Tensor] = None

            if step > 0 and step % 5 == 0:
                # We should get a loss at each episode end:
                assert all(obs.done), step  # Since batch_size == 1 for now.
                assert step_results is not None, (step, obs.task_labels)
                loss_tensor = step_results["loss"]
                loss: Loss = step_results["loss_object"]
                print(f"Loss at step {step}: {loss}")
                losses[step].append(loss)

            else:
                assert step_results is None

            print(
                f"Step {step}, episode {episodes}: x={obs.x}, done={obs.done}, task labels: {obs.task_labels}, loss_tensor: {loss_tensor}"
            )

            if step >= setting.train_max_steps:
                assert False, "Shouldn't the environment have closed at this point?"

    for step, step_losses in losses.items():
        print(f"Losses at step {step}:")
        for loss in step_losses:
            print(f"\t{loss}")
    # assert False, losses


@pytest.mark.parametrize(
    "input, expected",
    [
        (np.array([0, 0, 0, 0]), {0: np.arange(4)}),
        (torch.as_tensor([0, 0, 0, 0]), {0: torch.arange(4)}),
        (
            torch.as_tensor([0, 0, 1, 0]),
            {0: torch.LongTensor([0, 1, 3]), 1: torch.LongTensor([2])},
        ),
        (
            np.array([0, 0, 1, None]),
            {0: np.array([0, 1]), 1: np.array([2]), None: np.array([3])},
        ),
    ],
)
def test_get_task_indices(input, expected):
    actual = get_task_indices(input)
    assert str(actual) == str(expected)


@pytest.mark.parametrize(
    "indices",
    [
        slice(0, 10),  # all the same task (0)
        slice(0, 20),  # 10 from task 0, 10 from task 1
        slice(0, 30),  # 10 from task 0, 10 from task 1, 10 from task 2
        slice(0, 50),  # 10 from each task.
    ],
)
def test_task_inference_sl(
    mixed_samples: Dict[int, Tuple[Tensor, Tensor, Tensor]],
    indices: slice,
    config: Config,
):
    """ TODO: Write out a test that checks that when given a batch with data
    from different tasks, and when the model is multiheaded, it will use the
    right output head for each image.
    """
    # Get a mixed batch
    xs, ys, ts = map(torch.cat, zip(*mixed_samples.values()))
    xs = xs[indices]
    ys = ys[indices]
    ts = ts[indices].int()
    obs = ClassIncrementalSetting.Observations(x=xs, task_labels=None)

    setting = ClassIncrementalSetting()
    model = MultiHeadModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(batch_size=30, multihead=True),
        config=config,
    )

    class MockEncoder(nn.Module):
        def forward(self, x: Tensor):
            return x.new_ones([x.shape[0], model.hidden_size])

    mock_encoder = MockEncoder()
    model.encoder = mock_encoder

    for i in range(5):
        model.output_heads[str(i)] = MockOutputHead(
            input_space=spaces.Box(0, 1, [model.hidden_size]),
            action_space=spaces.Discrete(setting.action_space.n),
            Actions=setting.Actions,
            task_id=i,
        )
    model.output_head = model.output_heads["0"]

    forward_pass = model(obs)
    y_preds = forward_pass.actions.y_pred

    assert y_preds.shape == ts.shape
    # TODO: Check that the task inference works by changing the logits to be based on
    # the assigned task in the Mock output head.
    # assert torch.all(y_preds == ts * xs.view([xs.shape[0], -1]).mean(1))


@pytest.mark.timeout(120)
def test_task_inference_rl_easy(config: Config):
    from sequoia.methods.base_method import BaseMethod

    method = BaseMethod(config=config)
    from sequoia.settings.rl import IncrementalRLSetting

    setting = IncrementalRLSetting(
        dataset="cartpole",
        nb_tasks=2,
        max_episode_steps=20,
        train_max_steps=200,
        test_max_steps=200,
        config=config,
    )
    results = setting.apply(method)
    assert results
    # assert False, results.to_log_dict()


@pytest.mark.timeout(120)
def test_task_inference_rl_hard(config: Config):

    method = BaseMethod(config=config)

    setting = IncrementalRLSetting(
        dataset="cartpole",
        nb_tasks=2,
        train_max_steps=1000,
        test_max_steps=1000,
        config=config,
    )
    results = setting.apply(method)
    assert results
    # assert False, results.to_log_dict()


from sequoia.methods.base_method import BaseMethod
from sequoia.settings.sl import TraditionalSLSetting
from sequoia.settings.sl.continual.setting import subset


@pytest.mark.timeout(30)
def test_task_inference_multi_task_sl(config: Config):
    setting = TraditionalSLSetting(dataset="mnist", nb_tasks=2, config=config)
    # TODO: Maybe add this kind of 'max_steps_per_task' argument even in supervised
    # settings:
    dataset_length = 1000
    # TODO: Shorten the train/test datasets?
    method = BaseMethod(config=config, max_epochs=1)
    setting.setup()
    setting.train_datasets = [
        subset(dataset, list(range(dataset_length)))
        for dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        subset(dataset, list(range(dataset_length))) for dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        subset(dataset, list(range(dataset_length)))
        for dataset in setting.test_datasets
    ]

    results = setting.apply(method)
    assert 0.80 <= results.average_final_performance.objective
