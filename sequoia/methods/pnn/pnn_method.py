from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import gym
import numpy as np
import torch
import tqdm
from gym import spaces
from gym.spaces import Box
from numpy import inf
from simple_parsing import ArgumentParser
from wandb.wandb_run import Run

from sequoia.settings.base import Environment
from sequoia.common import Config
from sequoia.common.hparams import HyperParameters, categorical, log_uniform, uniform
from sequoia.common.spaces import Image
from sequoia.common.transforms.utils import is_image
from sequoia.methods import register_method
from sequoia.settings import (
    Actions,
    Method,
    Observations,
    PassiveEnvironment,
    RLSetting,
    Setting,
    TaskIncrementalRLSetting,
    TaskIncrementalSLSetting,
)
from sequoia.settings.assumptions import IncrementalAssumption
from sequoia.settings.assumptions.task_incremental import TaskIncrementalAssumption
from sequoia.utils import get_logger

from .model_rl import PnnA2CAgent
from .model_sl import PnnClassifier

logger = get_logger(__file__)

# BUG: Can't apply PNN to the ClassIncrementalSetting at the moment.
# BUG: Can't apply PNN to any RL Settings at the moment.
# (it was hard-coded to handle pixel cartpole).
# TODO: When those bugs get fixed, restore the 'IncrementalAssumption' as the target
# setting.
# TODO: Debugging PNN on Incremental rather than TaskIncremental


@register_method
class PnnMethod(Method, target_setting=IncrementalAssumption):
    """
    PNN Method.

    Applicable to both RL and SL Settings, as long as there are clear task boundaries
    during training (IncrementalAssumption).
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of the Pnn method. """

        # Learning rate of the optimizer. Defauts to 0.0001 when in SL.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=2e-4)
        num_steps: int = 200  # (only applicable in RL settings.)
        # Discount factor (Only used in RL settings).
        gamma: float = uniform(0.9, 0.999, default=0.99)
        # Number of hidden units (only used in RL settings.)
        hidden_size: int = categorical(64, 128, 256, default=256)
        # Batch size in SL, and number of parallel environments in RL.
        # Defaults to None in RL, and 32 when in SL.
        batch_size: Optional[int] = None
        # Maximum number of training epochs per task. (only used in SL Settings)
        max_epochs_per_task: int = uniform(1, 100, default=10)

    def __init__(self, hparams: HParams = None):
        # We will create those when `configure` will be called, before training.
        self.config: Optional[Config] = None
        self.task_id: Optional[int] = 0
        self.hparams: Optional[PnnMethod.HParams] = hparams
        self.model: Union[PnnA2CAgent, PnnClassifier]
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: Setting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """

        input_space: Box = setting.observation_space["x"]

        # For now all Settings have `Discrete` (i.e. classification) action spaces.
        action_space: spaces.Discrete = setting.action_space

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = action_space.n
        self.num_inputs = np.prod(input_space.shape)

        self.added_tasks = []
        if not (setting.task_labels_at_train_time and setting.task_labels_at_test_time): 
            logger.warning(RuntimeWarning(
                "TODO: PNN doesn't have 'propper' task inference, and task labels "
                "arent always available! This will use an output head at random."
            ))
        if isinstance(setting, RLSetting):
            # If we're applied to an RL setting:

            # Used these as the default hparams in RL:
            self.hparams = self.hparams or self.HParams()
            assert self.hparams
            self.train_steps_per_task = setting.steps_per_task

            # We want a batch_size of None, i.e. only one observation at a time.
            setting.batch_size = None

            self.num_steps = self.hparams.num_steps
            # Otherwise, we can train basically as long as we want on each task.
            self.loss_function = {
                "gamma": self.hparams.gamma,
            }
            if is_image(setting.observation_space.x):
                # Observing pixel input.
                self.arch = "conv"
            else:
                # Observing state input (e.g. the 4 floats in cartpole rather than images)
                self.arch = "mlp"
            self.model = PnnA2CAgent(self.arch, self.hparams.hidden_size)

        else:
            # If we're applied to a Supervised Learning setting:
            # Used these as the default hparams in SL:
            self.hparams = self.hparams or self.HParams(
                learning_rate=0.0001, batch_size=32,
            )
            if self.hparams.batch_size is None:
                self.hparams.batch_size = 32

            # Set the batch size on the setting.
            setting.batch_size = self.hparams.batch_size
            # For now all Settings on the supervised side of the tree have images as
            # inputs, so the observation spaces are of type `Image` (same as Box, but with
            # additional `h`, `w`, `c` and `b` attributes).
            assert isinstance(input_space, Image)
            assert (
                setting.increment == setting.test_increment
            ), "Assuming same number of classes per task for training and testing."
            # TODO: (@lebrice): Temporarily 'fixing' this by making it so each output
            # head has as many outputs as there are classes in total, which might make
            # no sense, but currently works.
            # It would be better to refactor this so that each output head can have only
            # as many outputs as is required, and then reshape / offset the predictions.
            n_outputs = setting.increment
            n_outputs = setting.action_space.n
            self.layer_size = [self.num_inputs, 256, n_outputs]
            self.model = PnnClassifier(n_layers=len(self.layer_size) - 1,)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting. """
        # This method gets called if task boundaries are known in the current
        # setting. Furthermore, if task labels are available, task_id will be
        # the index of the new task. If not, task_id will be None.
        # For example, you could do something like this:
        # self.model.current_task = task_id
        if self.training:
            self.model.freeze_columns([task_id])

        if task_id not in self.added_tasks:
            if isinstance(self.model, PnnA2CAgent):
                self.model.new_task(
                    device=self.device,
                    num_inputs=self.num_inputs,
                    num_actions=self.num_actions,
                )
            else:
                self.model.new_task(device=self.device, sizes=self.layer_size)

            self.added_tasks.append(task_id)

        self.task_id = task_id

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(self.task_id), lr=self.hparams.learning_rate,
        )

    def get_actions(
        self, observations: Observations, action_space: spaces.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for the given observations. """

        observations = observations.to(self.device)
        with torch.no_grad():
            if isinstance(self.model, PnnA2CAgent):
                predictions = self.model(observations)
                _, logit = predictions
                # get the predicted action:
                action = torch.argmax(logit).item()
            else:
                logits = self.model(observations)
                # Get the predicted classes
                y_pred = logits.argmax(dim=-1).cpu().numpy()
                action = y_pred

        assert action in action_space, (action, action_space)
        return action

    def fit(self, train_env: Environment, valid_env: Environment):
        """ Train and validate this method using the "environments" for the current task.

        NOTE: `train_env` and `valid_env` are both `gym.Env`s as well as `DataLoader`s.
        This means that if you want to write a "regular" SL training loop, you totally
        can, and if you want to write you RL-style training loop, you can also do that.
        """
        if isinstance(train_env.unwrapped, PassiveEnvironment):
            self.fit_sl(train_env, valid_env)
        else:
            self.fit_rl(train_env, valid_env)

    def fit_rl(self, train_env: gym.Env, valid_env: gym.Env):
        """ Training loop for Reinforcement Learning (a.k.a. "active") environment. """
        """
        base on https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
        """
        if self.model is None:
            self.model = PnnA2CAgent(self.arch, self.hparams.hidden_size)
        assert isinstance(self.model, PnnA2CAgent)

        self.set_optimizer()
        assert self.hparams
        # self.model.float()

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for episode in range(self.train_steps_per_task):
            values = []
            rewards = []
            log_probs = []

            state = train_env.reset()
            for steps in range(self.num_steps):
                value, policy_dist = self.model(state)

                value = value.item()
                dist = policy_dist.detach().numpy()

                action = np.random.choice(self.num_actions, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, done, _ = train_env.step(action)

                rewards.append(reward.y)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done or steps == self.num_steps - 1:
                    Qval, _ = self.model(state)
                    Qval = Qval.item()
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))

                    if episode % 10 == 0:
                        print(
                            f"episode: {episode}, "
                            f"reward: {np.sum(rewards)}, "
                            f"total length: {steps}, "
                            f"average length: {average_lengths[-1]}"
                        )
                    break

            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.hparams.gamma * Qval
                Qvals[t] = Qval

            # update actor critic
            values_tensor = torch.as_tensor(values, dtype=torch.float)
            Qvals = torch.as_tensor(Qvals, dtype=torch.float)
            log_probs_tensor = torch.stack(log_probs)

            advantage = Qvals - values_tensor
            actor_loss = (-log_probs_tensor * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

    def fit_sl(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Train on a Supervised Learning (a.k.a. "passive") environment. """
        observations: TaskIncrementalSLSetting.Observations = train_env.reset()
        cuda_observations = observations.to(self.device)
        assert isinstance(self.model, PnnClassifier)
        assert self.hparams

        self.set_optimizer()

        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.hparams.max_epochs_per_task):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with torch.set_grad_enabled(True), tqdm.tqdm(train_env) as train_pbar:
                postfix: Dict[str, Any] = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(
                        batch, environment=train_env,
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            with torch.set_grad_enabled(False), tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(
                        batch, environment=valid_env,
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> None:
        parser.add_arguments(cls.HParams, dest="hparams", default=None)

    @classmethod
    def from_argparse_args(cls, args: Namespace) -> "PnnMethod":
        hparams: PnnMethod.HParams = args.hparams
        method = cls(hparams=hparams)
        return method

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        return self.hparams.get_orion_space()

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.
        
        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        # Here we overwrite the corresponding attributes with the new suggested values
        # leaving other fields unchanged.
        # NOTE: These new hyper-paramers will be used in the next run in the sweep,
        # since each call to `configure` will create a new Model.
        self.hparams = self.hparams.replace(**new_hparams)

    def setup_wandb(self, run: Run) -> None:
        """ Called by the Setting when using Weights & Biases, after `wandb.init`.

        This method is here to provide Methods with the opportunity to log some of their
        configuration options or hyper-parameters to wandb.

        NOTE: The Setting has already set the `"setting"` entry in the `wandb.config` by
        this point.

        Parameters
        ----------
        run : wandb.Run
            Current wandb Run.
        """
        run.config["hparams"] = self.hparams.to_dict()


def main_rl():
    """ Applies the PnnMethod in a RL Setting. """
    parser = ArgumentParser(description=__doc__, add_dest_to_option_strings=False)

    Config.add_argparse_args(parser, dest="config")
    PnnMethod.add_argparse_args(parser, dest="method")

    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        nb_tasks=2,
        train_task_schedule={
            0: {"gravity": 10, "length": 0.3},
            1000: {"gravity": 10, "length": 0.5},
        },
    )

    args = parser.parse_args()

    config: Config = Config.from_argparse_args(args, dest="config")
    method: PnnMethod = PnnMethod.from_argparse_args(args, dest="method")
    method.config = config

    # 2. Creating the Method
    # method = ImproveMethod()

    # 3. Applying the method to the setting:
    results = setting.apply(method, config=config)

    print(results.summary())
    print(f"objective: {results.objective}")
    return results


def main_sl():
    """ Applies the PnnMethod in a SL Setting. """
    parser = ArgumentParser(description=__doc__, add_dest_to_option_strings=False)

    # Add arguments for the Setting
    # TODO: PNN is coded for the DomainIncrementalSetting, where the action space
    # is the same for each task.
    # parser.add_arguments(DomainIncrementalSetting, dest="setting")
    parser.add_arguments(TaskIncrementalSLSetting, dest="setting")
    # TaskIncrementalSLSetting.add_argparse_args(parser, dest="setting")
    Config.add_argparse_args(parser, dest="config")

    # Add arguments for the Method:
    PnnMethod.add_argparse_args(parser, dest="method")

    args = parser.parse_args()

    # setting: TaskIncrementalSLSetting = args.setting
    setting: TaskIncrementalSLSetting = TaskIncrementalSLSetting.from_argparse_args(
        # setting: DomainIncrementalSetting = DomainIncrementalSetting.from_argparse_args(
        args,
        dest="setting",
    )
    config: Config = Config.from_argparse_args(args, dest="config")

    method: PnnMethod = PnnMethod.from_argparse_args(args, dest="method")

    method.config = config

    results = setting.apply(method, config=config)
    print(results.summary())
    return results


if __name__ == "__main__":
    # Run RL Setting
    main_sl()
    # Run SL Setting
    # main_rl()
