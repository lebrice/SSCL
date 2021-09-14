""" Example of creating an A2C agent using the simplebaselines3 package.

See https://stable-baselines3.readthedocs.io/en/master/guide/install.html
"""
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Type, Union

import gym
import torch
from gym import spaces
from simple_parsing import choice, mutable_field
from stable_baselines3.common.base_class import (
    BaseAlgorithm,
    BasePolicy,
    DummyVecEnv,
    GymEnv,
    MaybeCallback,
    Monitor,
    VecEnv,
    VecTransposeImage,
    is_image_space,
    is_wrapped,
)
# from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from wandb.wandb_run import Run

from sequoia.common.gym_wrappers.batch_env.batched_vector_env import VectorEnv
from sequoia.common.gym_wrappers.utils import has_wrapper
from simple_parsing.helpers.hparams import HyperParameters, log_uniform, categorical
from sequoia.common.spaces import Image
from sequoia.common.transforms.utils import is_image
from sequoia.settings import Method, Setting
from sequoia.settings.rl.continual import ContinualRLSetting
from sequoia.settings.rl.wrappers import NoTypedObjectsWrapper, RemoveTaskLabelsWrapper
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.serialization import register_decoding_fn

logger = get_logger(__file__)

# "Patch" the _wrap_env function of the BaseAlgorithm class of
# stable_baselines, to make it recognize the VectorEnv from gym.vector as a
# vectorized environment.
# Stable-Baselines3 has a lot of duplicated code from openai gym


# def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
#     """ "
#     Wrap environment with the appropriate wrappers if needed.
#     For instance, to have a vectorized environment
#     or to re-order the image channels.

#     :param env:
#     :param verbose:
#     :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
#     :return: The wrapped environment.
#     """

#     # if not isinstance(env, VecEnv):
#     if not (
#         isinstance(env, (VecEnv, VectorEnv))
#         or isinstance(env.unwrapped, (VecEnv, VectorEnv))
#     ):
#         # if not is_wrapped(env, Monitor) and monitor_wrapper:
#         if monitor_wrapper and not (
#             is_wrapped(env, Monitor)
#             or is_wrapped(env, gym.wrappers.Monitor)
#             or has_wrapper(env, gym.wrappers.Monitor)
#         ):
#             if verbose >= 1:
#                 print("Wrapping the env with a `Monitor` wrapper")
#             env = Monitor(env)
#         if verbose >= 1:
#             print("Wrapping the env in a DummyVecEnv.")
#         env = DummyVecEnv([lambda: env])

#     if is_image_space(env.observation_space) and not is_wrapped(env, VecTransposeImage):
#         if verbose >= 1:
#             print("Wrapping the env in a VecTransposeImage.")
#         env = VecTransposeImage(env)

#     # check if wrapper for dict support is needed when using HER
#     if isinstance(env.observation_space, gym.spaces.dict.Dict):
#         env = ObsDictWrapper(env)

#     return env


# BaseAlgorithm._wrap_env = staticmethod(_wrap_env)


class RemoveInfoWrapper(gym.Wrapper):
    """ Wrapper used to remove the 'info' dict, since there seems to be a bug in sb3
    whenever there is something in the 'info' dict.
    """

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        info = {}
        return obs, rewards, done, info


@dataclass
class SB3BaseHParams(HyperParameters):
    """ Hyper-parameters of a model from the `stable_baselines3` package.

    The command-line arguments for these are created with simple-parsing.
    """

    # The policy model to use (MlpPolicy, CnnPolicy, ...)
    policy: Optional[Union[str, Type[BasePolicy]]] = choice(
        "MlpPolicy", "CnnPolicy", default=None
    )
    # # The base policy used by this method
    # policy_base: Type[BasePolicy]

    # learning rate for the optimizer, it can be a function of the current
    # progress remaining (from 1 to 0)
    learning_rate: Union[float, Callable] = log_uniform(1e-7, 1e-2, default=1e-4)
    # Additional arguments to be passed to the policy on creation
    policy_kwargs: Optional[Dict[str, Any]] = None
    # the log location for tensorboard (if None, no logging)
    tensorboard_log: Optional[str] = None
    # The verbosity level: 0 none, 1 training information, 2 debug
    verbose: int = 1
    # Device on which the code should run. By default, it will try to use a Cuda
    # compatible device and fallback to cpu if it is not possible.
    device: Union[torch.device, str] = "auto"

    # # Whether the algorithm supports training with multiple environments (as in A2C)
    # support_multi_env: bool = False

    # Whether to create a second environment that will be used for evaluating
    # the agent periodically. (Only available when passing string for the
    # environment)
    create_eval_env: bool = False

    # # When creating an environment, whether to wrap it or not in a Monitor wrapper.
    # monitor_wrapper: bool = True

    # Seed for the pseudo random generators
    seed: Optional[int] = None
    # # Whether to use generalized State Dependent Exploration (gSDE) instead of
    # action noise exploration (default: False)
    # use_sde: bool = False
    # # Sample a new noise matrix every n steps when using gSDE Default: -1
    # (only sample at the beginning of the rollout)
    # sde_sample_freq: int = -1

    # Wether to clear the experience buffer at the beginning of a new task.
    # NOTE: We use to_dict here so that it doesn't get passed do the Policy class.
    clear_buffers_between_tasks: bool = categorical(
        True, False, default=False, to_dict=False
    )


@dataclass
class StableBaselines3Method(Method, ABC, target_setting=ContinualRLSetting):
    """ Base class for the methods that use models from the stable_baselines3
    repo.
    """

    family: ClassVar[str] = "sb3"

    # Class variable that represents what kind of Model will be used.
    # (This is just here so we can easily create one Method class per model type
    # by just changing this class attribute.)
    Model: ClassVar[Type[BaseAlgorithm]]

    # HyperParameters of the Method.
    hparams: SB3BaseHParams = mutable_field(SB3BaseHParams)

    # The number of training steps to run per task.
    # NOTE: This shouldn't be set to more than the task length when applying this method
    # on a ContinualRLSetting, because we don't currently have a way of "resetting"
    # the nonstationarity in the environment, and there is only one task,
    # therefore if we trained for say 10 million steps, while the
    # non-stationarity only lasts for 10_000 steps, we'd have seen an almost
    # stationary distribution, since the environment would have stopped changing after
    # 10_000 steps.
    # train_steps_per_task: int = 10_000

    # callback(s) called at every step with state of the algorithm.
    callback: MaybeCallback = None
    # The number of timesteps before logging.
    log_interval: int = 100
    # the name of the run for TensorBoard logging
    tb_log_name: str = "run"
    # Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
    # TODO: Log the evaluations to wandb.
    eval_freq: int = 5_000
    # Number of episode to evaluate the agent
    n_eval_episodes = 5
    # Path to a folder where the evaluations will be saved
    eval_log_path: Optional[str] = None

    def __post_init__(self):
        self.model: Optional[BaseAlgorithm] = None
        # Extra wrappers to add to the train_env and valid_env before passing
        # them to the `learn` method from stable-baselines3.
        from sequoia.common.gym_wrappers import (
            TransformObservation,
            TransformAction,
            TransformReward,
        )
        import operator
        from functools import partial

        self.extra_train_wrappers: List[Callable[[gym.Env], gym.Env]] = [
            partial(TransformObservation, f=operator.itemgetter("x")),
            # partial(TransformAction, f=operator.itemgetter("y_pred"),
            partial(TransformReward, f=operator.itemgetter("y")),
            RemoveInfoWrapper,
        ]
        self.extra_valid_wrappers: List[Callable[[gym.Env], gym.Env]] = [
            partial(TransformObservation, f=operator.itemgetter("x")),
            partial(TransformReward, f=operator.itemgetter("y")),
            RemoveInfoWrapper,
        ]
        # Number of timesteps to train on for each task.
        self.total_timesteps_per_task: int = 0

        self.train_env: gym.Env = None
        self.valid_env: gym.Env = None

    def configure(self, setting: ContinualRLSetting):
        # Delete the model, if present.
        self.model = None
        # For now, we don't batch the space because stablebaselines3 will add an
        # additional batch dimension if we do.
        # TODO: Still need to debug the batching stuff with stablebaselines,
        # some methods support it, some don't, and it doesn't recognize
        # VectorEnvs from gym.
        setting.batch_size = None

        # BUG: Need to fix an issue when using the CnnPolicy and Atary envs, the
        # input shape isn't what they expect (only 2 channels instead of three
        # apparently.)
        # from sequoia.common.transforms import Transforms
        # NOTE: Important to not use any transforms, since the SB3 methods want to get
        # the 'raw' np.uint8 image as an input.
        transforms = [
            # Transforms.to_tensor,
            # Transforms.three_channels,
            # Transforms.channels_first_if_needed,
        ]
        setting.transforms = transforms
        setting.train_transforms = transforms
        setting.val_transforms = transforms
        setting.test_transforms = transforms

        if self.hparams.policy is None:
            if is_image(setting.observation_space.x):
                self.hparams.policy = "CnnPolicy"
            else:
                self.hparams.policy = "MlpPolicy"

        logger.debug(f"Will use {self.hparams.policy} as the policy.")
        # TODO: Double check that some settings might not impose a limit on
        # number of training steps per environment (e.g. task-incremental RL?)
        if setting.steps_per_phase:
            # if self.train_steps_per_task > setting.steps_per_phase:
            #     warnings.warn(
            #         RuntimeWarning(
            #             f"Can't train for the requested {self.train_steps_per_task} "
            #             f"steps, since we're (currently) only allowed a maximum of "
            #             f"{setting.steps_per_phase} steps.)"
            #         )
            #     )
            # Use as many training steps as possible.
            self.train_steps_per_task = setting.steps_per_phase - 1
        # Otherwise, we can train basically as long as we want on each task.

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        """ Create a Model given the training and validation environments. """
        model_kwargs = self.hparams.to_dict()
        assert "clear_buffers_between_tasks" not in model_kwargs
        return self.Model(env=train_env, **model_kwargs)

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        # Remove the extra information that the Setting gives us.
        for wrapper in self.extra_train_wrappers:
            train_env = wrapper(train_env)

        for wrapper in self.extra_valid_wrappers:
            valid_env = wrapper(valid_env)

        if self.model is None:
            self.model = self.create_model(train_env, valid_env)
        else:
            # TODO: "Adapt"/re-train the model on the new environment.
            # BUG: In the MT10 benchmark, the last entry in the observation space is
            # very slightly different, which prevents us from doing this:
            """
            >>> env.observation_space.low
            array([-0.525 ,  0.348 , -0.0525, -1.    ,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf, -0.525 ,  0.348 , -0.0525,
                    -1.,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf, -0.1   ,  0.8   ,  0.01  ], dtype=float32)
            >>> observation_space.low
            array([-0.525 ,  0.348 , -0.0525, -1.    ,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf, -0.525 ,  0.348 , -0.0525,
                    -1.,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,
                    -inf, -0.1   ,  0.8   ,  0.05  ], dtype=float32)
            """
            if self.train_env is not None:
                # BUG: MT10 has *slightly* different values in 'low' between tasks!
                if (
                    isinstance(train_env.observation_space, spaces.Box)
                    and train_env.observation_space.shape[-1] == 39
                ):
                    train_env.observation_space = self.train_env.observation_space
            self.model.set_env(train_env)
        self.train_env = train_env
        self.valid_env = valid_env

        # Decide how many steps to train on.
        total_timesteps = self.train_steps_per_task
        # TODO: Get the max number of steps directly from the env, rather than from the
        # setting's fields.
        logger.info(f"Starting training, for a maximum of {total_timesteps} steps.")
        # todo: Customize the parametrers of the model and/or of this "learn"
        # method if needed.
        self.model = self.model.learn(
            # The total number of samples (env steps) to train on
            total_timesteps=total_timesteps,
            eval_env=valid_env,
            callback=self.callback,
            log_interval=self.log_interval,
            tb_log_name=self.tb_log_name,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            eval_log_path=self.eval_log_path,
            # whether or not to reset the current timestep number (used in logging)
            reset_num_timesteps=True,
        )

    def get_actions(
        self, observations: ContinualRLSetting.Observations, action_space: spaces.Space
    ) -> ContinualRLSetting.Actions:
        obs = observations.x
        predictions = self.model.predict(obs)
        action, _ = predictions
        assert action in action_space, (observations, action, action_space)
        return action

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
        return {
            "algo_hparams": self.hparams.get_orion_space(),
        }

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
        self.hparams = self.hparams.replace(**new_hparams["algo_hparams"])

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

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting.

        If task labels are available, `task_id` will correspond to the index of
        the new task. Otherwise, if task labels aren't available, `task_id` will
        be `None`.

        todo: use this to customize how your method handles task transitions.
        """
        if self.hparams.clear_buffers_between_tasks:
            self.clear_buffers()

    def clear_buffers(self):
        """ Clears out the experience buffer of the Policy. """
        # I think that's the right way to do it.. not sure.
        # assert False, self.model.replay_buffer.pos
        if self.model:
            # TODO: These are really interesting methods!
            # self.model.save_replay_buffer
            # self.model.load_replay_buffer

            self.model.replay_buffer.reset()


# We do this just to prevent errors when trying to decode the hparams class above, and
# also to silence the related warnings from simple-parsing's decoding.py module.

register_decoding_fn(Type[BasePolicy], lambda v: v)
register_decoding_fn(Callable, lambda v: v)
