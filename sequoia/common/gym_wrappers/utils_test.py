import gym
from gym.wrappers import FilterObservation, ClipAction, AtariPreprocessing
from gym.wrappers.pixel_observation import PixelObservationWrapper
import pytest

from .pixel_observation import PixelObservationWrapper

from .utils import has_wrapper


@pytest.mark.parametrize("env,wrapper_type,result",
[
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), ClipAction, False),
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True),
    (lambda: PixelObservationWrapper(gym.make("CartPole-v0")), PixelObservationWrapper, True),
    # (AtariPreprocessing(gym.make("Breakout-v0")), ClipAction, True),
])
def test_has_wrapper(env, wrapper_type, result):
    assert has_wrapper(env(), wrapper_type) == result
