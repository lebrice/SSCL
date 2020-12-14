import gym
from gym import spaces
import pytest
import numpy as np
import torch
import PIL.Image

from . import (ChannelsFirst, ChannelsFirstIfNeeded, ChannelsLast, Compose,
               ThreeChannels, Transforms)


@pytest.mark.parametrize("transform,input_shape,output_shape",
[   
    ## Channels first:
    (Transforms.channels_first, (9, 9, 3), (3, 9, 9)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_first, (9, 12, 3), (3, 9, 12)),
    (Transforms.channels_first, (400, 600, 3), (3, 400, 600)),
    # Axes get permuted even when the channels are already 'first'.
    (Transforms.channels_first, (3, 12, 9), (9, 3, 12)),

    ## Channels first (if needed):
    (Transforms.channels_first_if_needed, (9, 9, 3), (3, 9, 9)),
    (Transforms.channels_first_if_needed, (9, 12, 3), (3, 9, 12)),
    (Transforms.channels_first_if_needed, (400, 600, 3), (3, 400, 600)),
    # Axes do NOT get permuted when the channels are already 'first'.
    (Transforms.channels_first_if_needed, (3, 12, 9), (3, 12, 9)),
    # Does nothing when the channel dim isn't in {1, 3}:
    (Transforms.channels_first_if_needed, (7, 12, 13), (7, 12, 13)),
    (Transforms.channels_first_if_needed, (7, 12, 123), (7, 12, 123)),
    # when the input is 4-dimensional with batch size of 1 or 3, still works:
    (Transforms.channels_first_if_needed, (1, 28, 12, 3), (1, 3, 28, 12)),
    (Transforms.channels_first_if_needed, (1, 400, 600, 3), (1, 3, 400, 600)),
    (Transforms.channels_first_if_needed, (1, 3, 28, 27), (1, 3, 28, 27)),

    (Transforms.channels_first_if_needed, (3, 28, 12, 3), (3, 3, 28, 12)),
    (Transforms.channels_first_if_needed, (3, 400, 600, 3), (3, 3, 400, 600)),
    (Transforms.channels_first_if_needed, (3, 3, 28, 27), (3, 3, 28, 27)),
    
    ## Channels Last:
    (Transforms.channels_last, (3, 9, 9), (9, 9, 3)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_last, (3, 9, 12), (9, 12, 3)),
    # Axes get permuted even when the channels are already 'last'.
    (Transforms.channels_last, (5, 6, 1), (6, 1, 5)),
    
    ## Channels Last (if needed):
    (Transforms.channels_last_if_needed, (3, 9, 9), (9, 9, 3)),
    # Check that the ordering doesn't get messed up:
    (Transforms.channels_last_if_needed, (3, 9, 12), (9, 12, 3)),
    # Axes do NOT get permuted when the channels are already 'last':
    (Transforms.channels_last_if_needed, (5, 6, 1), (5, 6, 1)),
    (Transforms.channels_last_if_needed, (12, 13, 3), (12, 13, 3)),
    # Does nothing when the channel dim isn't in {1, 3}:
    (Transforms.channels_last_if_needed, (7, 12, 13), (7, 12, 13)),
    
    # Test out the 'ThreeChannels' transform
    (Transforms.three_channels, (7, 12, 13), (7, 12, 13)),
    (Transforms.three_channels, (1, 28, 28), (3, 28, 28)),
    (Transforms.three_channels, (28, 28, 1), (28, 28, 3)),

    # Test out the 'Resize' transforms
    (Transforms.resize_64x64, (3, 128, 128), (3, 64, 64)),
    (Transforms.resize_64x64, (128, 128, 3), (64, 64, 3)),
    (Transforms.resize_64x64, (3, 64, 64), (3, 64, 64)),
    (Transforms.resize_64x64, (64, 64, 3), (64, 64, 3)),
    (Transforms.resize_64x64, (3, 111, 128), (3, 64, 64)),
    (Transforms.resize_64x64, (111, 128, 3), (64, 64, 3)),    
])
def test_transform(transform: Transforms, input_shape, output_shape):
    x = torch.rand(input_shape)
    y = transform(x)
    assert y.shape == output_shape
    assert y.shape == transform.shape_change(input_shape)
    
    input_space = spaces.Box(low=0, high=1, shape=input_shape)
    output_space = spaces.Box(low=0, high=1, shape=output_shape)
    
    actual_output_space = transform.space_change(input_space)
    assert actual_output_space == output_space

from PIL.Image import Image


@pytest.mark.parametrize("transform,input_shape,output_shape",
[   
    # NOTE: to_tensor also does the channels-first operation (because since the
    # torchvision transform ToTensor does it, we do it also).  
    (Transforms.to_tensor, (9, 9, 3), (3, 9, 9)),
    (Transforms.to_tensor, (3, 9, 9), (3, 9, 9)),
])
def test_to_tensor(transform: Transforms, input_shape, output_shape):
    x = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    # x = PIL.Image.fromarray(x, mode="RGB")
    y = transform(x)
    assert y.shape == output_shape
    assert y.shape == transform.shape_change(input_shape)
    assert isinstance(y, torch.Tensor)
    
    input_space = spaces.Box(low=0, high=255, shape=input_shape, dtype=np.uint8)
    output_space = spaces.Box(low=0, high=1, shape=output_shape, dtype=np.float32)
    
    actual_output_space = transform.space_change(input_space)
    assert actual_output_space == output_space


def test_compose_shape_change_same_as_result_shape():
    transform = Compose([Transforms.channels_first])
    start_shape = (9, 9, 3)
    x = transform(torch.rand(start_shape))
    assert x.shape == (3, 9, 9)
    assert x.shape == transform.shape_change(start_shape)
    assert x.shape == transform.shape_change(start_shape) == (3, 9, 9)

import gym
from sequoia.common.gym_wrappers import PixelObservationWrapper, TransformObservation


def test_channels_first_transform_on_gym_env():
    env = gym.make("CartPole-v0")
    env = PixelObservationWrapper(env)
    assert env.reset().shape == (400, 600, 3)
    
    transform = Compose([
        Transforms.to_tensor,
        Transforms.channels_first_if_needed,
    ])
    env = TransformObservation(env, transform)
    assert env.reset().shape == (3, 400, 600)
    assert env.observation_space.shape == (3, 400, 600)

    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (3, 400, 600)
