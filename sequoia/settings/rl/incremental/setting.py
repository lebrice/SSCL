import inspect
import json
import operator
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from itertools import islice
from io import StringIO
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

import gym
from gym import spaces
from sequoia.settings.base import Method
from sequoia.common.gym_wrappers import MultiTaskEnvironment, TransformObservation
from sequoia.common.gym_wrappers.utils import is_monsterkong_env
from sequoia.common.spaces import Sparse
from sequoia.common.metrics import EpisodeMetrics
from sequoia.common.transforms import Transforms
from sequoia.settings.assumptions.incremental import (
    IncrementalAssumption,
    TaskSequenceResults,
    TaskResults,
)
from sequoia.settings.base import Results
from sequoia.settings.rl.continual import ContinualRLSetting
from sequoia.settings.rl.envs import (
    ATARI_PY_INSTALLED,
    METAWORLD_INSTALLED,
    MONSTERKONG_INSTALLED,
    MTENV_INSTALLED,
    MUJOCO_INSTALLED,
    MetaMonsterKongEnv,
    MetaWorldEnv,
    MTEnv,
    MujocoEnv,
    metaworld_envs,
    mtenv_envs,
)
from sequoia.utils import constant, deprecated_property, dict_union, pairwise
from sequoia.utils.logging_utils import get_logger
from simple_parsing import field, list_field
from simple_parsing.helpers import choice
from typing_extensions import Final

from ..discrete.setting import DiscreteTaskAgnosticRLSetting
from ..discrete.setting import supported_envs as _parent_supported_envs
from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from .results import IncrementalRLResults
from .tasks import (
    EnvSpec,
    IncrementalTask,
    is_supported,
    make_incremental_task,
    sequoia_registry,
)

logger = get_logger(__file__)

# TODO: When we add a wrapper to 'concat' two environments, then we can move this
# 'passing custom env for each task' feature up into DiscreteTaskAgnosticRL.

supported_envs: Dict[str, EnvSpec] = dict_union(
    _parent_supported_envs,
    {
        spec.id: spec
        for env_id, spec in sequoia_registry.env_specs.items()
        if spec.id not in _parent_supported_envs and is_supported(env_id)
    },
)
if METAWORLD_INSTALLED:
    supported_envs["MT10"] = "MT10"
    supported_envs["MT50"] = "MT50"
    supported_envs["CW10"] = "CW10"
    supported_envs["CW20"] = "CW20"
available_datasets: Dict[str, str] = {env_id: env_id for env_id in supported_envs}


@dataclass
class IncrementalRLSetting(IncrementalAssumption, DiscreteTaskAgnosticRLSetting):
    """ Continual RL setting in which:
    - Changes in the environment's context occur suddenly (same as in Discrete, Task-Agnostic RL)
    - Task boundary information (and task labels) are given at training time
    - Task boundary information is given at test time, but task identity is not.
    """

    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    # The function used to create the tasks for the chosen env.
    _task_sampling_function: ClassVar[
        Callable[..., IncrementalTask]
    ] = make_incremental_task
    Results: ClassVar[Type[Results]] = IncrementalRLResults

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = available_datasets
    # Which dataset/environment to use for training, validation and testing.
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # # The number of tasks. By default 0, which means that it will be set
    # # depending on other fields in __post_init__, or eventually be just 1.
    # nb_tasks: int = field(0, alias=["n_tasks", "num_tasks"])

    # (Copied from the assumption, just for clarity:)
    # TODO: Shouldn't these kinds of properties be on the class, rather than on the
    # instance?

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: Final[bool] = constant(False)
    # Wether to give access to the task labels at train time.
    task_labels_at_train_time: Final[bool] = constant(True)
    # Wether to give access to the task labels at test time.
    task_labels_at_test_time: bool = False

    train_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()
    val_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()
    test_envs: List[Union[str, Callable[[], gym.Env]]] = list_field()

    def __post_init__(self):
        if self.dataset in ["MT10", "MT50", "CW10", "CW20"]:
            from metaworld import MT10, MT50, Task, MetaWorldEnv

            benchmarks = {
                "MT10": MT10,
                "MT50": MT50,
                "CW10": MT50,
                "CW20": MT50,
            }
            benchmark_class = benchmarks[self.dataset]
            logger.info(
                f"Creating metaworld benchmark {benchmark_class}, this might take a "
                f"while (~15 seconds)."
            )
            self._benchmark = benchmark_class(
                seed=self.config.seed if self.config else None
            )

            envs: Dict[str, Type[MetaWorldEnv]] = self._benchmark.train_classes
            env_tasks: Dict[str, List[Task]] = {
                env_name: [
                    task
                    for task in self._benchmark.train_tasks
                    if task.env_name == env_name
                ]
                for env_name, env_class in self._benchmark.train_classes.items()
            }
            train_env_tasks: Dict[str, List[Task]] = {}
            val_env_tasks: Dict[str, List[Task]] = {}
            test_env_tasks: Dict[str, List[Task]] = {}
            test_fraction = 0.1
            val_fraction = 0.1
            for env_name, env_tasks in env_tasks.items():
                n_tasks = len(env_tasks)
                n_val_tasks = int(max(1, n_tasks * val_fraction))
                n_test_tasks = int(max(1, n_tasks * test_fraction))
                n_train_tasks = len(env_tasks) - n_val_tasks - n_test_tasks
                if n_train_tasks <= 1:
                    # Can't create train, val and test tasks.
                    raise RuntimeError(
                        f"There aren't enough tasks for env {env_name} ({n_tasks}) "
                    )
                tasks_iterator = iter(env_tasks)
                train_env_tasks[env_name] = list(islice(tasks_iterator, n_train_tasks))
                val_env_tasks[env_name] = list(islice(tasks_iterator, n_val_tasks))
                test_env_tasks[env_name] = list(islice(tasks_iterator, n_test_tasks))
                assert train_env_tasks[env_name]
                assert val_env_tasks[env_name]
                assert test_env_tasks[env_name]

            max_train_steps_per_task = 1_000_000
            if self.train_steps_per_task is None:
                self.train_steps_per_task = max_train_steps_per_task

            if self.dataset in ["CW10", "CW20"]:
                # TODO: Raise a warning if the number of tasks is non-default and set to
                # something different than in the benchmark
                # Re-create the [ContinualWorld benchmark](@TODO: Add citation here)
                version = 2
                env_names = [
                    f"hammer-v{version}",
                    f"push-wall-v{version}",
                    f"faucet-close-v{version}",
                    f"push-back-v{version}",
                    f"stick-pull-v{version}",
                    f"handle-press-side-v{version}",
                    f"push-v{version}",
                    f"shelf-place-v{version}",
                    f"window-close-v{version}",
                    f"peg-unplug-side-v{version}",
                ]
                if self.train_steps_per_task > max_train_steps_per_task:
                    raise RuntimeError(
                        f"Can't use more than {max_train_steps_per_task} steps per "
                        f"task in the {self.dataset} benchmark!"
                    )

                # TODO: Decide the number of test steps.
                # NOTE: Should we allow using fewer steps?
                # NOTE: The default value for this field is 10_000 currently, so this
                # check doesn't do anything.
                if self.test_steps_per_task is None:
                    self.test_steps_per_task = 10_000

                if self.dataset == "CW20":
                    env_names = env_names * 2
                train_env_names = env_names
                val_env_names = env_names
                test_env_names = env_names
            else:
                train_env_names = list(train_env_tasks.keys())
                val_env_names = list(val_env_tasks.keys())
                test_env_names = list(test_env_tasks.keys())

            self.train_envs = [
                partial(
                    make_metaworld_env,
                    env_class=envs[env_name],
                    tasks=train_env_tasks[env_name],
                )
                for env_name in train_env_names
            ]
            self.val_envs = [
                partial(
                    make_metaworld_env,
                    env_class=envs[env_name],
                    tasks=val_env_tasks[env_name],
                )
                for env_name in val_env_names
            ]
            self.test_envs = [
                partial(
                    make_metaworld_env,
                    env_class=envs[env_name],
                    tasks=test_env_tasks[env_name],
                )
                for env_name in test_env_names
            ]

        # if is_monsterkong_env(self.dataset):
        #     if self.force_pixel_observations:
        #         # Add this to the kwargs that will be passed to gym.make, to make sure that
        #         # we observe pixels, and not state.
        #         self.base_env_kwargs["observe_state"] = False
        #     elif self.force_state_observations:
        #         self.base_env_kwargs["observe_state"] = True

        self._using_custom_envs_foreach_task: bool = False
        if self.train_envs:
            self._using_custom_envs_foreach_task = True
            # TODO: Raise a warning if we're going to overwrite a non-default nb_tasks?
            self.nb_tasks = len(self.train_envs)

            if not self.val_envs:
                # TODO: Use a wrapper that sets a different random seed?
                self.val_envs = self.train_envs.copy()
            if not self.test_envs:
                # TODO: Use a wrapper that sets a different random seed?
                self.test_envs = self.train_envs.copy()
            if (
                self.train_task_schedule
                or self.val_task_schedule
                or self.test_task_schedule
            ):
                raise RuntimeError(
                    "You can either pass `train/valid/test_envs`, or a task schedule, "
                    "but not both!"
                )

            self.train_dataset: Union[str, Callable[[], gym.Env]] = self.train_envs[0]
            self.val_dataset: Union[str, Callable[[], gym.Env]] = self.val_envs[0]
            self.test_dataset: Union[str, Callable[[], gym.Env]] = self.test_envs[0]
        else:
            if self.val_envs or self.test_envs:
                raise RuntimeError(
                    "Can't pass `val_envs` or `test_envs` without passing `train_envs`."
                )

        super().__post_init__()

        if self._using_custom_envs_foreach_task:
            # TODO: Use 'no-op' task schedules for now.
            # self.train_task_schedule.clear()
            # self.val_task_schedule.clear()
            # self.test_task_schedule.clear()
            pass

            # TODO: Check that all the envs have the same observation spaces!
            # (If possible, find a way to check this without having to instantiate all
            # the envs.)

        # TODO: If the dataset has a `max_path_length` attribute, then it's probably
        # a Mujoco / metaworld / etc env, and so we set a limit on the episode length to
        # avoid getting an error.
        max_path_length: Optional[int] = getattr(
            self._temp_train_env, "max_path_length", None
        )
        if self.max_episode_steps is None and max_path_length is not None:
            assert max_path_length > 0
            self.max_episode_steps = max_path_length

        # if self.dataset == "MetaMonsterKong-v0":
        #     # TODO: Limit the episode length in monsterkong?
        #     # TODO: Actually end episodes when reaching a task boundary, to force the
        #     # level to change?
        #     self.max_episode_steps = self.max_episode_steps or 500

        # FIXME: Really annoying little bugs with these three arguments!
        # self.nb_tasks = self.max_steps // self.steps_per_task

    @property
    def train_task_lengths(self) -> List[int]:
        """ Gives the length of each training task (in steps for now). """
        return [
            task_b_step - task_a_step
            for task_a_step, task_b_step in pairwise(
                sorted(self.train_task_schedule.keys())
            )
        ]

    @property
    def train_phase_lengths(self) -> List[int]:
        """ Gives the length of each training 'phase', i.e. the maximum number of (steps
        for now) that can be taken in the training environment, in a single call to .fit
        """
        return [
            task_b_step - task_a_step
            for task_a_step, task_b_step in pairwise(
                sorted(self.train_task_schedule.keys())
            )
        ]

    @property
    def current_train_task_length(self) -> int:
        """ Deprecated field, gives back the max number of steps per task. """
        if self.stationary_context:
            return sum(self.train_task_lengths)
        return self.train_task_lengths[self.current_task_id]

    # steps_per_task: int = deprecated_property(
    #     "steps_per_task", "current_train_task_length"
    # )
    # @property
    # def steps_per_task(self) -> int:
    #     # unique_task_lengths = list(set(self.train_task_lengths))
    #     warning = DeprecationWarning(
    #         "The 'steps_per_task' attribute is deprecated, use "
    #         "`current_train_task_length` instead, which gives the length of the "
    #         "current task."
    #     )
    #     warnings.warn(warning)
    #     logger.warning(warning)
    #     return self.current_train_task_length

    # @property
    # def steps_per_phase(self) -> int:
    #     # return unique_task_lengths
    #     test_task_lengths: List[int] = [
    #         task_b_step - task_a_step
    #         for task_a_step, task_b_step in pairwise(
    #             sorted(self.test_task_schedule.keys())
    #         )
    #     ]

    @property
    def task_label_space(self) -> gym.Space:
        # TODO: Explore an alternative design for the task sampling, based more around
        # gym spaces rather than the generic function approach that's currently used?
        # IDEA: Might be cleaner to put this in the assumption class
        task_label_space = spaces.Discrete(self.nb_tasks)
        if not self.task_labels_at_train_time or not self.task_labels_at_test_time:
            sparsity = 1
            if self.task_labels_at_train_time ^ self.task_labels_at_test_time:
                # We have task labels "50%" of the time, ish:
                sparsity = 0.5
            task_label_space = Sparse(task_label_space, sparsity=sparsity)
        return task_label_space

    def setup(self, stage: str = None) -> None:
        # Called before the start of each task during training, validation and
        # testing.
        super().setup(stage=stage)
        # What's done in ContinualRLSetting:
        # if stage in {"fit", None}:
        #     self.train_wrappers = self.create_train_wrappers()
        #     self.valid_wrappers = self.create_valid_wrappers()
        # elif stage in {"test", None}:
        #     self.test_wrappers = self.create_test_wrappers()
        if self._using_custom_envs_foreach_task:
            logger.debug(
                f"Using custom environments from `self.[train/val/test]_envs` for task "
                f"{self.current_task_id}."
            )
            # NOTE: Here is how this supports passing custom envs for each task: We just
            # switch out the value of this property, and let the
            # `train/val/test_dataloader` methods work as usual!
            # TODO: Instantiate the envs if needed.
            # TODO: Add support for batching these environments:
            def instantiate_all_envs(envs: List[Union[Callable[[], gym.Env], gym.Env]]) -> List[gym.Env]:
                return [
                    env() if not isinstance(env, gym.Env) else env for env in envs
                ]
            self.train_envs = instantiate_all_envs(self.train_envs)
            self.val_envs = instantiate_all_envs(self.val_envs)
            self.test_envs = instantiate_all_envs(self.test_envs)

            
            if self.stationary_context:
                from sequoia.settings.rl.discrete.multienv_wrappers import RandomMultiEnvWrapper, ConcatEnvsWrapper
                
               
                self.train_dataset = RandomMultiEnvWrapper(self.train_envs, add_task_ids=self.task_labels_at_train_time)
                self.val_dataset = RandomMultiEnvWrapper(self.val_envs, add_task_ids=self.task_labels_at_train_time)
                self.test_dataset = ConcatEnvsWrapper(self.test_envs, add_task_ids=self.task_labels_at_test_time)
            elif self.known_task_boundaries_at_train_time:
                self.train_dataset = self.train_envs[self.current_task_id]
                self.val_dataset = self.val_envs[self.current_task_id]
                # TODO: The test loop goes through all the envs, hence this doesn't really
                # work.
                self.test_dataset = self.test_envs[self.current_task_id]
            else:
                self.train_dataset = ConcatEnvsWrapper(self.train_envs, add_task_ids=self.task_labels_at_train_time)
                self.val_dataset = ConcatEnvsWrapper(self.val_envs, add_task_ids=self.task_labels_at_train_time)
                self.test_dataset = ConcatEnvsWrapper(self.test_envs, add_task_ids=self.task_labels_at_test_time)
            # Check that the observation/action spaces are all the same for all
            # the train/valid/test envs
            # NOTE: Skipping this check for now.
            # self._check_all_envs_have_same_spaces(
            #     envs_or_env_functions=self.train_envs, wrappers=self.train_wrappers,
            # )
            # # TODO: Inconsistent naming between `val_envs` and `valid_wrappers` etc.
            # self._check_all_envs_have_same_spaces(
            #     envs_or_env_functions=self.val_envs, wrappers=self.valid_wrappers,
            # )
            # self._check_all_envs_have_same_spaces(
            #     envs_or_env_functions=self.test_envs, wrappers=self.test_wrappers,
            # )
        else:
            # TODO: Should we populate the `self.train_envs`, `self.val_envs` and
            # `self.test_envs` fields here as well, just to be consistent?
            # base_env = self.dataset
            # def task_env(task_index: int) -> Callable[[], MultiTaskEnvironment]:
            #     return self._make_env(
            #         base_env=base_env,
            #         wrappers=[],
            #     )
            # self.train_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # self.val_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # self.test_envs = [partial(gym.make, self.dataset) for i in range(self.nb_tasks)]
            # assert False, self.train_task_schedule
            pass

    # def _setup_fields_using_temp_env(self, temp_env: MultiTaskEnvironment):
    #     """ Setup some of the fields on the Setting using a temporary environment.

    #     This temporary environment only lives during the __post_init__() call.
    #     """
    #     super()._setup_fields_using_temp_env(temp_env)
    def test_dataloader(
        self, batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ):
        if not self._using_custom_envs_foreach_task:
            return super().test_dataloader(
                batch_size=batch_size, num_workers=num_workers
            )

        # raise NotImplementedError("TODO:")

        # IDEA: Pretty hacky, but might be cleaner than adding fields for the moment.
        test_max_steps = self.test_max_steps
        test_max_episodes = self.test_max_episodes
        self.test_max_steps = test_max_steps // self.nb_tasks
        if self.test_max_episodes:
            self.test_max_episodes = test_max_episodes // self.nb_tasks
        # self.test_env = self.TestEnvironment(self.test_envs[self.current_task_id])

        task_test_env = super().test_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )

        self.test_max_steps = test_max_steps
        self.test_max_episodes = test_max_episodes
        return task_test_env

    def test_loop(self, method: Method["IncrementalRLSetting"]):
        if not self._using_custom_envs_foreach_task:
            return super().test_loop(method)

        # TODO: If we're using custom envs for each task, then the test loop needs to be
        # re-organized.
        # raise NotImplementedError(
        #     f"TODO: Need to add a wrapper that can switch between envs, or "
        #     f"re-write the test loop."
        # )
        assert self.nb_tasks == len(self.test_envs), "assuming this for now."
        test_envs = []
        for task_id in range(self.nb_tasks):
            # TODO: Make sure that self.test_dataloader() uses the right number of steps
            # per test task (current hard-set to self.test_max_steps).
            task_test_env = self.test_dataloader()
            test_envs.append(task_test_env)
        from ..discrete.multienv_wrappers import ConcatEnvsWrapper

        on_task_switch_callback = getattr(method, "on_task_switch", None)
        joined_test_env = ConcatEnvsWrapper(
            test_envs,
            add_task_ids=self.task_labels_at_test_time,
            on_task_switch_callback=on_task_switch_callback,
        )
        # TODO: Use this 'joined' test environment in this test loop somehow.
        # IDEA: Hacky way to do it: (I don't think this will work as-is though)
        _test_dataloader_method = self.test_dataloader
        self.test_dataloader = lambda *args, **kwargs: joined_test_env
        super().test_loop(method)
        self.test_dataloader = _test_dataloader_method

        test_loop_results = DiscreteTaskAgnosticRLSetting.Results()
        for task_id, test_env in enumerate(test_envs):
            # TODO: The results are still of the wrong type, because we aren't changing
            # the type of test environment or the type of Results
            results_of_wrong_type: IncrementalRLResults = test_env.get_results()
            # For now this weird setup means that there will be only one 'result'
            # object in this that actually has metrics:
            # assert results_of_wrong_type.task_results[task_id].metrics
            all_metrics: List[EpisodeMetrics] = sum(
                [result.metrics for result in results_of_wrong_type.task_results], []
            )
            n_metrics_in_each_result = [
                len(result.metrics) for result in results_of_wrong_type.task_results
            ]
            # assert all(n_metrics == 0 for i, n_metrics in enumerate(n_metrics_in_each_result) if i != task_id), (n_metrics_in_each_result, task_id)
            # TODO: Also transfer the other properties like runtime, online performance,
            # etc?
            # TODO: Maybe add addition for these?
            # task_result = sum(results_of_wrong_type.task_results)
            task_result = TaskResults(metrics=all_metrics)
            # task_result: TaskResults[EpisodeMetrics] = results_of_wrong_type.task_results[task_id]
            test_loop_results.task_results.append(task_result)
        return test_loop_results

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        In this Incremental-RL Setting, fit is called once per task.
        (Same as ClassIncrementalSetting in SL).
        """
        return self.nb_tasks

    @staticmethod
    def _make_env(
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        wrappers: List[Callable[[gym.Env], gym.Env]] = None,
        **base_env_kwargs: Dict,
    ) -> gym.Env:
        """ Helper function to create a single (non-vectorized) environment.

        This is also used to create the env whenever `self.dataset` is a string that
        isn't registered in gym. This happens for example when using an environment from
        meta-world (or mtenv).
        """
        # Check if the env is registed in a known 'third party' gym-like package, and if
        # needed, create the base env in the way that package requires.
        if isinstance(base_env, str):
            env_id = base_env

            # Check if the id belongs to mtenv
            if MTENV_INSTALLED and env_id in mtenv_envs:
                from mtenv import make as mtenv_make

                # This is super weird. Don't undestand at all
                # why they are doing this. Makes no sense to me whatsoever.
                base_env = mtenv_make(env_id, **base_env_kwargs)

                # Add a wrapper that will remove the task information, because we use
                # the same MultiTaskEnv wrapper for all the environments.
                wrappers.insert(0, MTEnvAdapterWrapper)

            if METAWORLD_INSTALLED and env_id in metaworld_envs:
                # TODO: Should we use a particular benchmark here?
                # For now, we find the first benchmark that has an env with this name.
                import metaworld

                for benchmark_class in [metaworld.ML10]:
                    benchmark = benchmark_class()
                    if env_id in benchmark.train_classes.keys():
                        # TODO: We can either let the base_env be an env type, or
                        # actually instantiate it.
                        base_env: Type[MetaWorldEnv] = benchmark.train_classes[env_id]
                        # NOTE: (@lebrice) Here I believe it's better to just have the
                        # constructor, that way we re-create the env for each task.
                        # I think this might be better, as I don't know for sure that
                        # the `set_task` can be called more than once in metaworld.
                        # base_env = base_env_type()
                        break
                else:
                    raise NotImplementedError(
                        f"Can't find a metaworld benchmark that uses env {env_id}"
                    )

        return ContinualRLSetting._make_env(
            base_env=base_env, wrappers=wrappers, **base_env_kwargs,
        )

    def create_task_schedule(
        self, temp_env: gym.Env, change_steps: List[int], seed: int = None,
    ) -> Dict[int, Dict]:
        task_schedule: Dict[int, Dict] = {}
        if self._using_custom_envs_foreach_task:
            # If custom envs were passed to be used for each task, then we don't create
            # a "task schedule", because the only reason we're using a task schedule is
            # when we want to change something about the 'base' env in order to get
            # multiple tasks.
            # Create a task schedule dict, just to fit in?
            for i, task_step in enumerate(change_steps):
                task_schedule[task_step] = {}
            return task_schedule

        # TODO: Make it possible to use something other than steps as keys in the task
        # schedule, something like a NamedTuple[int, DeltaType], e.g. Episodes(10) or
        # Steps(10), something like that!
        # IDEA: Even fancier, we could use a TimeDelta to say "do one hour of task 0"!!
        for step in change_steps:
            # TODO: Add a `stage` argument (an enum or something with 'train', 'valid'
            # 'test' as values, and pass it to this function. Tasks should be the same
            # in train/valid for now, given the same task Id.
            # TODO: When the Results become able to handle a different ordering of tasks
            # at train vs test time, allow the test task schedule to have different
            # ordering than train / valid.
            task = type(self)._task_sampling_function(
                temp_env,
                step=step,
                change_steps=change_steps,
                seed=seed,
            )
            task_schedule[step] = task

        return task_schedule

    def create_train_wrappers(self):
        # TODO: Clean this up a bit?
        if self._using_custom_envs_foreach_task:
            # TODO: Maybe do something different here, since we don't actually want to
            # add a CL wrapper at all in this case?
            assert not any(self.train_task_schedule.values())
            base_env = self.train_envs[self.current_task_id]
        else:
            base_env = self.train_dataset
        # assert False, super().create_train_wrappers()
        if self.stationary_context:
            task_schedule_slice = self.train_task_schedule.copy()
            assert len(task_schedule_slice) >= 2
            # Need to pop the last task, so that we don't sample it by accident!
            max_step = max(task_schedule_slice)
            last_task = task_schedule_slice.pop(max_step)
            # TODO: Shift the second-to-last task to the last step
            last_boundary = max(task_schedule_slice)
            second_to_last_task = task_schedule_slice.pop(last_boundary)
            task_schedule_slice[max_step] = second_to_last_task
            if 0 not in task_schedule_slice:
                assert self.nb_tasks == 1
                task_schedule_slice[0] = second_to_last_task
            # assert False, (max_step, last_boundary, last_task, second_to_last_task)
        else:
            current_task = list(self.train_task_schedule.values())[self.current_task_id]
            task_length = self.train_max_steps // self.nb_tasks
            task_schedule_slice = {
                0: current_task,
                task_length: current_task,
            }
        return self._make_wrappers(
            base_env=base_env,
            task_schedule=task_schedule_slice,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.transforms + self.train_transforms,
            starting_step=0,
            max_steps=max(task_schedule_slice.keys()),
            new_random_task_on_reset=self.stationary_context,
        )

    def create_valid_wrappers(self):
        if self._using_custom_envs_foreach_task:
            # TODO: Maybe do something different here, since we don't actually want to
            # add a CL wrapper at all in this case?
            assert not any(self.val_task_schedule.values())
            base_env = self.val_envs[self.current_task_id]
        else:
            base_env = self.val_dataset
        # assert False, super().create_train_wrappers()
        if self.stationary_context:
            task_schedule_slice = self.val_task_schedule
        else:
            current_task = list(self.val_task_schedule.values())[self.current_task_id]
            task_length = self.train_max_steps // self.nb_tasks
            task_schedule_slice = {
                0: current_task,
                task_length: current_task,
            }
        return self._make_wrappers(
            base_env=base_env,
            task_schedule=task_schedule_slice,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.transforms + self.val_transforms,
            starting_step=0,
            max_steps=max(task_schedule_slice.keys()),
            new_random_task_on_reset=self.stationary_context,
        )

    def create_test_wrappers(self):
        if self._using_custom_envs_foreach_task:
            # TODO: Maybe do something different here, since we don't actually want to
            # add a CL wrapper at all in this case?
            assert not any(self.test_task_schedule.values())
            base_env = self.test_envs[self.current_task_id]
        else:
            base_env = self.test_dataset
        # assert False, super().create_train_wrappers()
        task_schedule_slice = self.test_task_schedule
        # if self.stationary_context:
        # else:
        #     current_task = list(self.test_task_schedule.values())[self.current_task_id]
        #     task_length = self.test_max_steps // self.nb_tasks
        #     task_schedule_slice = {
        #         0: current_task,
        #         task_length: current_task,
        #     }
        return self._make_wrappers(
            base_env=base_env,
            task_schedule=task_schedule_slice,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.transforms + self.test_transforms,
            starting_step=0,
            max_steps=self.test_max_steps,
            new_random_task_on_reset=self.stationary_context,
        )

    def _check_all_envs_have_same_spaces(
        self,
        envs_or_env_functions: List[Union[str, gym.Env, Callable[[], gym.Env]]],
        wrappers: List[Callable[[gym.Env], gym.Wrapper]],
    ) -> None:
        """ Checks that all the environments in the list have the same
        observation/action spaces.
        """
        first_env = self._make_env(
            base_env=envs_or_env_functions[0], wrappers=wrappers, **self.base_env_kwargs
        )
        first_env.close()
        for task_id, task_env_id_or_function in zip(
            range(1, len(envs_or_env_functions)), envs_or_env_functions[1:]
        ):
            task_env = self._make_env(
                base_env=task_env_id_or_function,
                wrappers=wrappers,
                **self.base_env_kwargs,
            )
            task_env.close()
            if task_env.observation_space != first_env.observation_space:
                raise RuntimeError(
                    f"Env at task {task_id} doesn't have the same observation "
                    f"space ({task_env.observation_space}) as the environment of "
                    f"the first task: {first_env.observation_space}."
                )
            if task_env.action_space != first_env.action_space:
                raise RuntimeError(
                    f"Env at task {task_id} doesn't have the same action "
                    f"space ({task_env.action_space}) as the environment of "
                    f"the first task: {first_env.action_space}"
                )

    def _make_wrappers(
        self,
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        task_schedule: Dict[int, Dict],
        # sharp_task_boundaries: bool,
        task_labels_available: bool,
        transforms: List[Transforms],
        starting_step: int,
        max_steps: int,
        new_random_task_on_reset: bool,
    ) -> List[Callable[[gym.Env], gym.Env]]:
        if self._using_custom_envs_foreach_task:
            if any(task_schedule.values()):
                logger.warning(
                    RuntimeWarning(
                        f"Ignoring task schedule {task_schedule}, since custom envs were "
                        f"passed for each task!"
                    )
                )
            task_schedule = None

        wrappers = super()._make_wrappers(
            base_env=base_env,
            task_schedule=task_schedule,
            task_labels_available=task_labels_available,
            transforms=transforms,
            starting_step=starting_step,
            max_steps=max_steps,
            new_random_task_on_reset=new_random_task_on_reset,
        )

        if self._using_custom_envs_foreach_task:
            # If the user passed a specific env to use for each task, then there won't
            # be a MultiTaskEnv wrapper in `wrappers`, since the task schedule is
            # None/empty.
            # Instead, we will add a Wrapper that always gives the task ID of the
            # current task.

            # TODO: There are some 'unused' args above: `starting_step`, `max_steps`,
            # `new_random_task_on_reset` which are still passed to the super() call, but
            # just unused.
            if new_random_task_on_reset:
                pass
                # raise NotImplementedError(
                #     "TODO: Add a MultiTaskEnv wrapper of some sort that alternates "
                #     " between the source envs."
                # )
            else:
                assert not task_schedule
                task_label = self.current_task_id
                task_label_space = spaces.Discrete(self.nb_tasks)
                if not task_labels_available:
                    task_label = None
                    task_label_space = Sparse(task_label_space, sparsity=1.0)

                wrappers.append(
                    partial(
                        AddTaskIDWrapper,
                        task_label=task_label,
                        task_label_space=task_label_space,
                    )
                )

        if is_monsterkong_env(base_env):
            # TODO: Need to register a MetaMonsterKong-State-v0 or something like that!
            # TODO: Maybe add another field for 'force_state_observations' ?
            # if self.force_pixel_observations:
            pass

        return wrappers


class MTEnvAdapterWrapper(TransformObservation):
    # TODO: For now, we remove the task id portion of the space and of the observation
    # dicts.
    def __init__(self, env: MTEnv, f: Callable = operator.itemgetter("env_obs")):
        super().__init__(env=env, f=f)
        # self.observation_space = self.env.observation_space["env_obs"]

    # def observation(self, observation):
    #     return observation["env_obs"]


from typing import Any

from sequoia.common.gym_wrappers.multi_task_environment import add_task_labels
from sequoia.common.gym_wrappers.utils import IterableWrapper


class AddTaskIDWrapper(IterableWrapper):
    """ Wrapper that adds always the same given task id to the observations.

    Used when the list of envs for each task is passed, so that each env also has the
    task id as part of their observation space and in their observations.
    """

    def __init__(
        self, env: gym.Env, task_label: Optional[int], task_label_space: gym.Space
    ):
        super().__init__(env=env)
        self.task_label = task_label
        self.task_label_space = task_label_space
        self.observation_space = add_task_labels(
            self.env.observation_space, task_labels=task_label_space
        )

    def observation(self, observation: Union[IncrementalRLSetting.Observations, Any]):
        return add_task_labels(observation, self.task_label)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.observation(obs), reward, done, info


def make_metaworld_env(
    env_class: Type[MetaWorldEnv], tasks: List["Task"]
) -> MetaWorldEnv:
    env = env_class()
    env.set_task(tasks[0])
    env = MultiTaskEnvironment(
        env,
        task_schedule={
            i: operator.methodcaller("set_task", task) for i, task in enumerate(tasks)
        },
        new_random_task_on_reset=True,
        add_task_dict_to_info=False,
        add_task_id_to_obs=False,
    )
    return env
