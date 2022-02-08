""" Handlers for creating tasks in different environments.

TODO: Add more envs:
- [ ] PyBullet!
- [ ] Box2d!
- [ ] ProcGen!
- [ ] dm_control!

from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore
"""
import difflib
import inspect
import warnings
from functools import partial, singledispatch
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

import gym
import numpy as np
from gym.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    Continuous_MountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)
from gym.envs.registration import EnvRegistry, EnvSpec, load, registry, spec
from sequoia.common.gym_wrappers.multi_task_environment import make_env_attributes_task
from sequoia.settings.rl.envs import MUJOCO_INSTALLED, EnvVariantSpec, sequoia_registry
from sequoia.utils.utils import camel_case

# Idea: Create a true 'Task' class?
Task = Any
ContinuousTask = Dict[str, float]
TaskType = TypeVar("TaskType", bound=ContinuousTask)
# TODO: Create a fancier class for the TaskSchedule, as described in the test file.
# IDEA: Have the Task Schedule be a 'list' of Task objects, each of which has a
# 'duration' parameter, which are accumulated to create the 'keys' of the task schedule!
# TaskSchedule = Dict[int, TaskType]


class TaskSchedule(Dict[int, TaskType]):
    pass


class EnvironmentNotSupportedError(gym.error.UnregisteredEnv):
    """ Error raised when we don't know how to create a task for the given environment.
    """


def names_match(name_a: str, name_b: str) -> bool:
    a_variants = (name_a, name_a.lower(), camel_case(name_a))
    b_variants = (name_b, name_b.lower(), camel_case(name_b))
    # TODO: Not sure about this 'endswith' stuff, e.g. with MountainCarContinuous vs MountainCar?
    return (
        name_a in b_variants or name_b in a_variants
    )  # or name_a.endswith(b_variants) or name_b.endswith(a_variants)


def _is_supported(
    env_id: str,
    _make_task_function: Callable[..., ContinuousTask],
    env_registry: EnvRegistry = registry,
) -> bool:
    """ Returns wether Sequoia is able to create (continuous) tasks for the given
    environment.

    WIP: It is better not to use this directly, and instead use the equivalent
    `is_supported` function which is created dynamically below.
    """

    def _has_handler(some_env_type: Type[gym.Env]) -> bool:
        """ Returns wether the "make task" function has a registered handler for the
        given envs.
        """
        return some_env_type in _make_task_function.registry or (
            not inspect.isfunction(some_env_type)
            and _make_task_function.dispatch(some_env_type)
            is not _make_task_function.dispatch(object)
        )

    if isinstance(env_id, str):
        env_spec = env_registry.spec(env_id)

    elif isinstance(env_id, EnvSpec):
        env_spec = env_id
        env_id = env_spec.id

    elif inspect.isclass(env_id) and issubclass(env_id, gym.Env):
        env_type = env_id
        env_spec = None
        if _has_handler(env_type):
            return True
        env_id = env_type.__name__
        class_name = env_type.__name__
    else:
        raise NotImplementedError(env_id, type(env_id))

    assert isinstance(env_id, str)
    if env_spec:
        assert isinstance(env_spec, EnvSpec)

        if callable(env_spec.entry_point):
            if _has_handler(env_spec.entry_point):
                return True
            class_name = env_spec.entry_point.__name__
        else:
            assert isinstance(env_spec.entry_point, str)
            _module, _, class_name = env_spec.entry_point.partition(":")

    registered_class_names = tuple(c.__name__ for c in _make_task_function.registry)

    if class_name in registered_class_names:
        return True
    elif class_name.startswith(registered_class_names):
        return True

    close_matches = difflib.get_close_matches(class_name, registered_class_names)
    if not close_matches:
        return False
    return False


def task_sampling_function(
    env_registry: EnvRegistry = registry, based_on: Callable[[gym.Env], TaskType] = None
) -> Callable[[gym.Env], TaskType]:
    """ Decorator for a "make_task" function (e.g. `make_continuous_task`,
    `make_discrete_task`, etc.) that does the following:
    
    1. Creates a singledispatch callable from the given function, if necessary;
    2. Registers three useful handlers, for strings, environment types, and wrappers to
    the new function.
    3. Adds a 'is_supported' function on that function (see NOTE below);
    4. Adds all the registered handlers from the `based_on` function, if passed;

    NOTE (@lebrice): not sure about this is_supported being created and set on the
    function itself. It would probably be cleaner to create a class like TaskCreator or
    something that has the same methods as the underlying singledispatch callable.

    NOTE: A task sampling function should give back the same task when given the same
    seed, step and change_steps.
    """

    def _wrapper(
        make_task_fn: Callable[[gym.Env], TaskType]
    ) -> Callable[[gym.Env], TaskType]:

        if not hasattr(make_task_fn, "registry"):
            make_task_fn = singledispatch(make_task_fn)

        @make_task_fn.register(type)
        def make_discrete_task_from_type(
            env_type: Type[gym.Env], **kwargs
        ) -> ContinuousTask:
            try:
                # Try to create a task without actually instantiating the env, by passing the
                # type of env as the 'env' argument, rather than an env instance.
                env_handler_function = make_task_fn.dispatch(env_type)
                return env_handler_function(env_type, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to create a task based only on the env type {env_type}: {exc}\n"
                ) from exc

        @make_task_fn.register(str)
        def make_discrete_task_by_id(env: str, **kwargs,) -> Union[Dict[str, Any], Any]:
            # Load the entry-point class, and use it to determine what handler to use.
            # TODO: Actually instantiate the env here? or just dispatch based on the env class?
            if env not in env_registry.env_specs:
                raise RuntimeError(
                    f"Can't create a task for env id {env}, since it isn't a registered env id."
                )
            env_spec: EnvSpec = env_registry.env_specs[env]
            env_entry_point: Callable[..., gym.Env] = load(env_spec.entry_point)
            # import inspect

            try:
                task: ContinuousTask = make_discrete_task_from_type(
                    env_entry_point, **kwargs
                )
                return task

            except RuntimeError as exc:
                warnings.warn(
                    RuntimeWarning(
                        f"A temporary environment will have to be created in order to make a task: {exc}"
                    )
                )

            with gym.make(env) as temp_env:
                # IDEA: Could avoid re-creating the env between calls to this function, for
                # instance by saving a single temp env in a global variable and overwriting
                # it if `env` is of a different type.
                return make_task_fn(temp_env, **kwargs)

        @make_task_fn.register
        def make_discrete_for_wrapped_env(
            env: gym.Wrapper, step: int, change_steps: List[int] = None, **kwargs,
        ) -> Union[Dict[str, Any], Any]:
            # NOTE: Not sure if this is totally a good idea...
            # If someone registers a handler for some kind of Wrapper, than all envs wrapped
            # with that wrapper will use that handler, instead of their base environment type.
            return make_task_fn(env.env, step=step, change_steps=change_steps, **kwargs)

        if based_on is not None:
            for registered_type, registered_handler in based_on.registry.items():
                # NOTE: Skipping these types since we register new handlers above. Not
                # sure if it's necessary, since it might just overwrite an old handler
                # to register a new one for the same type?
                if registered_type not in [object, str, type, gym.Wrapper]:
                    make_task_fn.register(registered_type, registered_handler)

        make_task_fn.is_supported = partial(_is_supported, _make_task_fn=make_task_fn)

        return make_task_fn

    return _wrapper


@singledispatch
def make_continuous_task(
    env: gym.Env, step: int, change_steps: List[int], seed: int = None, **kwargs,
) -> ContinuousTask:
    """ Generic function used by Sequoia's RL settings to create a "task" that will be
    applied to an environment like `env`.
    
    To add support for a new type of environment, simply register a handler function:

    ```
    @make_continuous_task.register(SomeGymEnvClass)
    def make_task_for_my_env(env: SomeGymEnvClass, step: int, change_steps: List[int], **kwargs,):
        return {"my_attribute": random.random()}
    ```

    NOTE: In order to create tasks for an environment through its string 'id', and to
    avoid having to actually instantiate an environment, `env` could perhaps be a type
    of environment rather than an actual environment instance. If your function can't
    handle this (raises an exception somehow), then a temporary environment will be
    created, and a warning will be raised.

    TODO: remove / rename this 'change_steps' to 'max_steps' instead.
    """
    raise NotImplementedError(f"Don't currently know how to create tasks for env {env}")


make_continuous_task = task_sampling_function(env_registry=sequoia_registry)(
    make_continuous_task
)
is_supported = partial(_is_supported, _make_task_function=make_continuous_task)

# from functools import _SingleDispatchCallable

# Dictionary mapping from environment type to a dict of environment values which can be
# modified with multiplicative gaussian noise.
_ENV_TASK_ATTRIBUTES: Dict[Union[Type[gym.Env]], Dict[str, float]] = {
    CartPoleEnv: {
        "gravity": 9.8,
        "masscart": 1.0,
        "masspole": 0.1,
        "length": 0.5,
        "force_mag": 10.0,
        "tau": 0.02,
    },
    PendulumEnv: {
        "max_speed": 8.0,
        "max_torque": 2.0,
        # "dt" = .05
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
    },
    MountainCarEnv: {
        "gravity": 0.0025,
        "goal_position": 0.45,  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        # BUG: Since we use multiplicative noise, this won't change over time.
        # "goal_velocity": 0,
    },
    Continuous_MountainCarEnv: {
        "goal_position": 0.45,  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        # BUG: Since we use multiplicative noise, this won't change over time.
        # "goal_velocity": 0,
    },
    # TODO: Test AcrobotEnv
    AcrobotEnv: {
        "LINK_LENGTH_1": 1.0,  # [m]
        "LINK_LENGTH_2": 1.0,  # [m]
        "LINK_MASS_1": 1.0,  #: [kg] mass of link 1
        "LINK_MASS_2": 1.0,  #: [kg] mass of link 2
        "LINK_COM_POS_1": 0.5,  #: [m] position of the center of mass of link 1
        "LINK_COM_POS_2": 0.5,  #: [m] position of the center of mass of link 2
        "LINK_MOI": 1.0,  #: moments of inertia for both links
    },
    # TODO: Add more of the classic control envs here.
    # TODO: Need to get the attributes to modify in each environment type and
    # add them here.
    # AtariEnv: [
    #     # TODO: Maybe have something like the difficulty as the CL 'task' ?
    #     # difficulties = temp_env.ale.getAvailableDifficulties()
    #     # "game_difficulty",
    # ],
}


@make_continuous_task.register(CartPoleEnv)
@make_continuous_task.register(PendulumEnv)
@make_continuous_task.register(MountainCarEnv)
@make_continuous_task.register(Continuous_MountainCarEnv)
@make_continuous_task.register(AcrobotEnv)
def make_task_for_classic_control_env(
    env: gym.Env,
    step: int,
    change_steps: List[int] = None,
    task_params: Union[List[str], Dict[str, Any]] = None,
    seed: int = None,
    noise_std: float = 0.2,
):
    # NOTE: `step` doesn't matter here, all tasks are independant.
    task_params = task_params or _ENV_TASK_ATTRIBUTES[type(env.unwrapped)]
    if step == 0:
        # Use the 'default' task as the first task.
        return task_params.copy()

    # Make this more reproducible: When given the same seed and same step, return the
    # same task.
    if seed is not None:
        rng = np.random.default_rng(seed + step)
    else:
        rng = None
    # Default back to the 'env attributes' task, which multiplies the default values
    # with normally distributed scaling coefficients.
    # TODO: Need to refactor the whole MultiTaskEnv/SmoothTransition wrappers / tasks
    # etc.
    return make_env_attributes_task(
        env, task_params=task_params, rng=rng, noise_std=noise_std,
    )


# IDEA: Could probably not have these big ugly IF statements since we have the stubs for
# the different mujoco env classes anyway.

if MUJOCO_INSTALLED:
    from sequoia.settings.rl.envs.mujoco import (
        ContinualHalfCheetahV2Env,
        ContinualHalfCheetahV3Env,
        ContinualHopperV2Env,
        ContinualHopperV3Env,
        ContinualWalker2dV2Env,
        ContinualWalker2dV3Env,
        ModifiedGravityEnv,
    )

    default_mujoco_gravity = -9.81

    @make_continuous_task.register(ContinualHopperV2Env)
    @make_continuous_task.register(ContinualHopperV3Env)
    @make_continuous_task.register(ContinualWalker2dV2Env)
    @make_continuous_task.register(ContinualWalker2dV3Env)
    @make_continuous_task.register(ContinualHalfCheetahV2Env)
    @make_continuous_task.register(ContinualHalfCheetahV3Env)
    def make_task_for_modified_gravity_env(
        env: ModifiedGravityEnv,
        step: int,
        change_steps: List[int],
        seed: int = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        step_seed = seed * step if seed is not None else None
        # NOTE: np.random.default_rng(None) will NOT give the same result every first
        # time it is called, so this won't cause any issues with the same gravity being
        # sampled for all tasks if `seed` is None.
        rng = np.random.default_rng(step_seed)
        if step == 0:
            coefficient = 1
        else:
            coefficient = rng.uniform() + 0.5
        # TODO: Do we want to start with normal gravity?
        gravity = coefficient * default_mujoco_gravity
        return {"gravity": gravity}

