from typing import Callable, List, TypeVar, Union, Tuple, Optional, Sequence

import gym
import torch
from gym import spaces
from torch import Tensor
from torchvision.transforms import Compose as ComposeBase

from sequoia.utils.logging_utils import get_logger
from .transform import Transform, InputType, OutputType
logger = get_logger(__file__)

T = TypeVar("T", bound=Callable)


class Compose(List[T], ComposeBase, Transform[InputType, OutputType]):
    """ Extend the Compose class of torchvision with methods of `list`.
    
    This can also be passed in members of the `Transforms` enum, which makes it
    possible to do something like this:
    >>> from .transform_enum import Compose, Transforms
    >>> transforms = Compose([Transforms.to_tensor, Transforms.three_channels,])
    >>> Transforms.three_channels in transforms
    True
    >>> transforms += [Transforms.random_grayscale]
    >>> transforms
    [<Transforms.to_tensor: ToTensor()>, <Transforms.three_channels: ThreeChannels()>, <Transforms.random_grayscale: RandomGrayscale(p=0.1)>]

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ComposeBase.__init__(self, transforms=self)
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        logger.debug(f"shape_change on Compose: input shape: {input_shape}")
        # TODO: Give the impact of this transform on a given input shape.
        for transform in self:
            logger.debug(f"Shape before transform {transform}: {input_shape}")
            shape_change_method: Optional[Callable] = getattr(transform, "shape_change", None)
            if shape_change_method and callable(shape_change_method):
                input_shape = transform.shape_change(input_shape)  # type: ignore
            else:
                logger.debug(
                    f"Unable to detect the change of shape caused by "
                    f"transform {transform}, assuming its output has same "
                    f"shape as its input."
                )
        logger.debug(f"Final shape: {input_shape}")
        return input_shape

    def space_change(self, input_space: gym.Space) -> gym.Space:
        from .transform_enum import Transforms
        for transform in self:
            if isinstance(transform, Transforms):
                transform = transform.value
            if hasattr(transform, "space_change"):
                input_space = transform.space_change(input_space)
        return input_space