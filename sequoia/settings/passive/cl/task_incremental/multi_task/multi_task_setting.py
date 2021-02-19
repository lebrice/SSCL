import itertools
from dataclasses import dataclass
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

import tqdm
from simple_parsing import list_field
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sequoia.common.gym_wrappers import RenderEnvWrapper, TransformObservation
from sequoia.settings.assumptions.incremental import TestEnvironment
from sequoia.settings.base import Method
from sequoia.settings.passive.cl.class_incremental_setting import \
    ClassIncrementalTestEnvironment
from sequoia.settings.passive.cl.objects import Observations, Actions, Rewards
from sequoia.settings.passive.passive_environment import PassiveEnvironment
from sequoia.utils import get_logger
from sequoia.utils.utils import constant
from ..task_incremental_setting import TaskIncrementalSetting

logger = get_logger(__file__)


@dataclass
class MultiTaskSetting(TaskIncrementalSetting):
    """IID version of the Task-Incremental Setting, where the data is shuffled.
    
    Can be used to estimate the upper bound performance of Task-Incremental CL Methods.
    """

    # Number of tasks.
    nb_tasks: int = 0

    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes.
    increment: Union[int, List[int]] = list_field(
        2, type=int, nargs="*", alias="n_classes_per_task"
    )
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = 0
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = constant(None)
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = constant(None)
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = constant(None)
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = constant(None)

    def __post_init__(self):
        super().__post_init__()
        # IDEA: We could reuse the training loop from Incremental, if we modify it so it
        # discriminates between "phases" and "tasks".

    @property
    def phases(self) -> int:
        return 1

    def setup(self, stage=None, *args, **kwargs):
        super().setup(stage=stage, *args, **kwargs)

    def train_loop(self, method: Method):
        """Runs a multi-task training loop, wether in RL or CL.
        
        IDEA: We could probably do it the same way in both RL and SL:
        1. Create the 'datasets' for all the tasks;
        2. "concatenate"+"Shuffle" the "datasets":
            - in SL: ConcatDataset / shuffle the datasets
            - in RL: Create a true `MultiTaskEnvironment` that accepts a list of envs as
              an input and alternates between environments at each episode.
              (either round-robin style, or randomly)
        """
        logger.info(f"Starting training")
        # Creating the dataloaders ourselves (rather than passing 'self' as
        # the datamodule)
        # TODO: Pass the train_dataloader and val_dataloader methods, rather than the envs?
        task_train_loader = self.train_dataloader()
        task_valid_loader = self.val_dataloader()
        success = method.fit(train_env=task_train_loader, valid_env=task_valid_loader,)
        task_train_loader.close()
        task_valid_loader.close()
    
    def get_train_dataset(self) -> Dataset:
        return ConcatDataset(self.train_datasets)

    def get_val_dataset(self) -> Dataset:
        return ConcatDataset(self.val_datasets)

    def get_test_dataset(self) -> Dataset:
        return ConcatDataset(self.test_datasets)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> PassiveEnvironment:
        """Returns a DataLoader for the training dataset.

        This dataloader will yield batches which will very likely contain data from
        multiple different tasks, and will contain task labels.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to use. Defaults to None, in which case the value of
            `self.batch_size` is used.
        num_workers : int, optional
            Number of workers to use. Defaults to None, in which case the value of
            `self.num_workers` is used.

        Returns
        -------
        PassiveEnvironment
            A "Passive" Dataloader/gym.Env. 
        """
        return super().train_dataloader(batch_size=batch_size, num_workers=num_workers)

    def val_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset.

        This dataloader will yield batches which will very likely contain data from
        multiple different tasks, and will contain task labels.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to use. Defaults to None, in which case the value of
            `self.batch_size` is used.
        num_workers : int, optional
            Number of workers to use. Defaults to None, in which case the value of
            `self.num_workers` is used.

        Returns
        -------
        PassiveEnvironment
            A "Passive" Dataloader/gym.Env. 
        """
        return super().val_dataloader(batch_size=batch_size, num_workers=num_workers)

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> PassiveEnvironment:
        """Returns a DataLoader for the test dataset.

        This dataloader will yield batches which will very likely contain data from
        multiple different tasks, and will contain task labels.

        Unlike the train and validation environments, the test environment will not
        yield rewards until the action has been sent to it using either `send` (when
        iterating in the DataLoader-style) or `step` (when interacting with the
        environment in the gym.Env style). For more info, take a look at the
        `PassiveEnvironment` class.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size to use. Defaults to None, in which case the value of
            `self.batch_size` is used.
        num_workers : int, optional
            Number of workers to use. Defaults to None, in which case the value of
            `self.num_workers` is used.

        Returns
        -------
        PassiveEnvironment
            A "Passive" Dataloader/gym.Env. 
        """
        return super().test_dataloader(batch_size=batch_size, num_workers=num_workers)

    def test_loop(self, method: Method) -> "IncrementalSetting.Results":
        """ Runs a multi-task test loop and returns the Results.
        """
        test_env = self.test_dataloader()
        try:
            # If the Method has `test` defined, use it.
            method.test(test_env)
            test_env.close()
            # Get the metrics from the test environment
            test_results: Results = test_env.get_results()
            print(f"Test results: {test_results}")
            return test_results

        except NotImplementedError:
            logger.info(
                f"Will query the method for actions at each step, "
                f"since it doesn't implement a `test` method."
            )

        obs = test_env.reset()

        # TODO: Do we always have a maximum number of steps? or of episodes?
        # Will it work the same for Supervised and Reinforcement learning?
        max_steps: int = getattr(test_env, "step_limit", None)

        # Reset on the last step is causing trouble, since the env is closed.
        pbar = tqdm.tqdm(itertools.count(), total=max_steps, desc="Test")
        episode = 0
        for step in pbar:
            if test_env.is_closed():
                logger.debug(f"Env is closed")
                break
            # logger.debug(f"At step {step}")
            action = method.get_actions(obs, test_env.action_space)

            # logger.debug(f"action: {action}")
            # TODO: Remove this:
            if isinstance(action, Actions):
                action = action.y_pred
            if isinstance(action, Tensor):
                action = action.cpu().numpy()

            obs, reward, done, info = test_env.step(action)

            if done and not test_env.is_closed():
                # logger.debug(f"end of test episode {episode}")
                obs = test_env.reset()
                episode += 1

        test_env.close()
        test_results = test_env.get_results()

        return test_results
