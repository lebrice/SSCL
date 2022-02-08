"""TODO: A Wrapper that creates smooth transitions between tasks.
Could be based on the MultiTaskEnvironment, but with a moving average update of
the task, rather than setting a brand new random task.

There could also be some kind of 'task_duration' parameter, and the model does
linear or smoothed-out transitions between them depending on the step number?
"""
from functools import singledispatch
from typing import Dict, List, Optional, TypeVar

import gym
import numpy as np
from gym import spaces

from torch import Tensor
from sequoia.common.spaces.sparse import Sparse
from sequoia.utils.logging_utils import get_logger

from .multi_task_environment import MultiTaskEnvironment, add_task_labels

logger = get_logger(__file__)


## TODO (@lebrice): Really cool idea!: Create a TaskSchedule class that inherits
# from Dict and when you __getitem__ a missing key, returns an interpolation!


class SmoothTransitions(MultiTaskEnvironment):
    """ Extends MultiTaskEnvironment to support smooth task boudaries.

    Same as `MultiTaskEnvironment`, but when in between two tasks, the
    environment will have its values set to a linear interpolation of the
    attributes from the two neighbouring tasks.
    ```
    env = gym.make("CartPole-v0")
    env = SmoothTransitions(env, task_schedule={
        10: dict(length=1.0),
        20: dict(length=2.0),
    })
    env.seed(123)
    env.reset()
    ```

    At step 0, the length is the default value (0.5)
    at step 1, the length is 0.5 + (1 / 10) * (1.0-0.5) = 0.55
    at step 2, the length is 0.5 + (2 / 10) * (1.0-0.5) = 0.60,
    etc.

    NOTE: This only works with float attributes at the moment.

    """

    def __init__(
        self,
        env: gym.Env,
        add_task_dict_to_info: bool = False,
        add_task_id_to_obs: bool = False,
        only_update_on_episode_end: bool = False,
        nb_tasks: int = None,
        **kwargs
    ):
        """ Wraps the environment, allowing for smooth task transitions.

        Same as `MultiTaskEnvironment`, but when in between two tasks, the
        environment will have its values set to a linear interpolation of the
        attributes from the two neighbouring tasks.


        TODO: Should we update the task paramers only on resets? or at each
        step? Might save a little bit of compute to only do it on resets, but
        then it's not exactly as 'smooth' as we would like it to be, especially
        if a single episode can be very long!

        NOTE: Assumes that the attributes are floats for now.

        Args:
            env (gym.Env): The gym environment to wrap.
            task_schedule (Dict[int, Dict[str, float]], optional) (Same as
                `MultiTaskEnvironment`): Dict mapping from a given step
                number to the attributes to be set at that time. Interpolations
                between the two neighbouring tasks will be used between task
                transitions.
            only_update_on_episode_end (bool, optional): When `False` (default),
                update the attributes of the environment smoothly after each
                step. When `True`, only update at the end of episodes (when
                `reset()` is called).
        """
        super().__init__(
            env,
            add_task_dict_to_info=add_task_dict_to_info,
            add_task_id_to_obs=add_task_id_to_obs,
            nb_tasks=nb_tasks,
            **kwargs
        )
        self.only_update_on_episode_end: bool = only_update_on_episode_end
        if self._max_steps is None and len(self.task_schedule) > 1:
            # TODO: DO we want to prevent going past the 'task step' in the task schedule?
            pass

        if isinstance(self.env.unwrapped, gym.vector.VectorEnv):
            raise NotImplementedError(
                "This isn't really supposed to be applied on top of a "
                "vectorized environment, rather, it should be used within each"
                " individual env."
            )

        if self.add_task_id_to_obs:
            nb_tasks = nb_tasks if nb_tasks is not None else len(self.task_schedule)
            self.observation_space = add_task_labels(
                self.env.observation_space,
                Sparse(spaces.Discrete(n=nb_tasks), sparsity=1.0),
            )

    def step(self, *args, **kwargs):
        if not self.only_update_on_episode_end:
            self.smooth_update()
        results = super().step(*args, **kwargs)
        return results

    def reset(self, **kwargs):
        # TODO: test this out.
        if self.only_update_on_episode_end:
            self.smooth_update()
        return super().reset(**kwargs)

    @property
    def current_task_id(self) -> Optional[int]:
        """ Returns the 'index' of the current task within the task schedule.

        In this case, we return None, since there aren't clear task boundaries.
        """
        return None

    def task_array(self, task: Dict[str, float]) -> np.ndarray:
        return np.array([task.get(k, self.default_task[k]) for k in self.task_params])

    def smooth_update(self) -> None:
        """ Update the curren_task at every step, based on a smooth mix of the
        previous and the next task. Every time we reach a _step that is in the
        task schedule, we update the 'prev_task_step' and 'next_task_step'
        attributes.
        """

        current_task: Dict[str, float] = {}
        for attr in self.task_params:
            steps: List[int] = []
            # list of the
            fixed_points: List[float] = []
            for step, task in sorted(self.task_schedule.items()):
                steps.append(step)
                fixed_points.append(task.get(attr, self.default_task[attr]))
            # logger.debug(f"{attr}: steps={steps}, fp={fixed_points}")
            interpolated_value: float = np.interp(
                x=self.steps, xp=steps, fp=fixed_points,
            )
            current_task[attr] = interpolated_value
            # logger.debug(f"interpolated value of {attr} at step {self.step}: {interpolated_value}")
        # logger.debug(f"Updating task at step {self.step}: {current_task}")
        self.current_task = current_task

