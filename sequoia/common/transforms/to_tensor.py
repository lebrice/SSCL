""" Slight modification of the ToTensor transform from TorchVision.

@lebrice: I wrote this because I would often get weird 'negative stride in
images' errors when converting PIL images from some gym environments when
using `ToTensor` from torchvision.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import Callable, Dict, Sequence, Tuple, TypeVar, Union, overload

import gym
import numpy as np
import torch
from gym import Space, spaces
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToTensor as ToTensor_
from torchvision.transforms import functional as F

from sequoia.common.gym_wrappers.convert_tensors import (add_tensor_support,
                                                         has_tensor_support)
from sequoia.common.spaces import NamedTuple, NamedTupleSpace, TypedDictSpace
from sequoia.utils import singledispatchmethod
from sequoia.utils.generic_functions import to_tensor
from sequoia.utils.logging_utils import get_logger

from .transform import Img, Transform
from .channels import channels_first_if_needed
logger = get_logger(__file__)


def copy_if_negative_strides(image: Img) -> Img:
    # It sometimes happens when taking images from a gym env that the strides
    # are negative, for some reason. Therefore we need to copy the array
    # before we can call torchvision.transforms.functional.to_tensor(image).
    if isinstance(image, Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        strides = image.strides
    elif isinstance(image, Tensor):
        strides = image.stride()
    elif hasattr(image, "strides"):
        strides = image.strides
    else:
        raise NotImplementedError(f"Can't get strides of object {image}")
    if any(s < 0 for s in strides):
        return image.copy()
    return image


@singledispatch
def image_to_tensor(image: Union[Img, Sequence[Img], gym.Space]) -> Union[Tensor, gym.Space]:
    """
    Converts a PIL Image or numpy.ndarray ((N) x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape ((N) x C x H x W) in the range
    [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F,
    RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

    Parameters
    ----------
    image : Union[Img, Sequence[Img]]
        [description]

    Returns
    -------
    Tensor
        [description]
    """
    raise NotImplementedError(f"Don't know how to convert {image} to a Tensor.")

# @image_to_tensor.register
# def _(image: Tensor) -> Tensor:
#     return channels_first_if_needed(image)

@image_to_tensor.register(Tensor)
@image_to_tensor.register(np.ndarray)
@image_to_tensor.register(Image)
def _(image: Union[Image, np.ndarray]) -> Tensor:
    """ Converts a PIL Image, or np.uint8 ndarray to a Tensor. Also reshapes it
    to channels_first format (because ToTensor from torchvision does it also).
    """    
    from .channels import (channels_first_if_needed, channels_last_if_needed,
                           has_channels_first, has_channels_last)
    image = copy_if_negative_strides(image)

    if len(image.shape) == 2:
        return F.to_tensor(image)

    if isinstance(image, np.ndarray):
        # Convert to channels last if needed, because ToTensor expects to
        # receive that.
        image = channels_first_if_needed(image)
        image = torch.from_numpy(image).contiguous()
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            image = image.float().div(255)
        return image


    if len(image.shape) == 4:
        return channels_first_if_needed(
            torch.stack(list(map(image_to_tensor, image)))
        )

    if not isinstance(image, Tensor):
        image = F.to_tensor(image)
    return channels_first_if_needed(image)


@image_to_tensor.register(list)
def _list_of_images_to_tensor(image: Sequence[Img]) -> Tensor:
    return torch.stack(list(map(image_to_tensor, image)))


@image_to_tensor.register(tuple)
def _to_tensor_effect_on_image_shape(image: Tuple[int, ...]) -> Tuple[int, ...]:
    """ Give the output shape given the input shape of an image. """
    if len(image) == 3:        
        from .channels import channels_first_if_needed
        return channels_first_if_needed(image)
    return image


@image_to_tensor.register(spaces.Box)
def _(image: spaces.Box) -> spaces.Box:
    if image.dtype == np.uint8:
        # images get their bounds changed to [0. 1.] and their shape changed to
        # channels_first.
        image = type(image)(low=0., high=1., shape=channels_first_if_needed(image.shape), dtype=np.float32)
    # TODO: it sometimes happens that the `image` space has already been
    # through 'to_tensor`, not sure what to do in that case.
    # elif not has_tensor_support(image):
    #     raise RuntimeError(f"image spaces should have dtype np.uint8!: {image}")
    # Since the transform would convert images / ndarrays to tensors, then we
    # add 'Tensor' support when applying the same transform on the Space of
    # images!
    image = add_tensor_support(image)
    return image


@image_to_tensor.register(NamedTupleSpace)
def _(space: Dict, device: torch.device = None) -> Dict:
    from .resize import is_image
    return type(space)(**{
        key: image_to_tensor(value) if is_image(value) else value 
        for key, value in space.items()
    }, dtype=space.dtype)


@image_to_tensor.register(Mapping)
@image_to_tensor.register(spaces.Dict)
def _space_with_images_to_tensor(space: Dict, device: torch.device = None) -> Dict:
    from .resize import is_image
    return type(space)(**{
        key: image_to_tensor(value) if is_image(value) else value 
        for key, value in space.items()
    })


@image_to_tensor.register(TypedDictSpace)
def _space_with_images_to_tensor(space: TypedDictSpace, device: torch.device = None) -> TypedDictSpace:
    from .resize import is_image
    return type(space)({
        key: image_to_tensor(value) if is_image(value) else value 
        for key, value in space.items()
    }, dtype=space.dtype)




# @image_to_tensor.register(Image)
# def to_tensor(image: Union[Img, Sequence[Img]]) -> Tensor:
    
#     tensor: Tensor
#     if isinstance(image, Tensor):
#         return channels_first(image)
#         return image
#         # return channels_first(image)

#     if isinstance(image, (list, tuple)) or (isinstance(image, np.ndarray) and image.ndim == 4):
#         return torch.stack(list(map(to_tensor, image)))

#     assert isinstance(image, (np.ndarray, Image))
#     image = copy_if_negative_strides(image)

#     if isinstance(image, np.ndarray):
#         # Convert to channels last if needed, because ToTensor expects to
#         # receive that.
#         if len(image.shape) == 2:
#             pass
#         elif image.shape[-1] not in {1, 3}:
#             assert image.shape[0] in {1, 3}, image.shape
#             image = image.transpose(1, 2, 0)
#         # image = channels_last(image)
#     image = F.to_tensor(image)
#     assert isinstance(image, Tensor), image.shape
#     return image


@dataclass
class ToTensor(ToTensor_, Transform):
    def __call__(self, image):
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
        
        Returns:
            Tensor: Converted image.
        
        NOTE: torchvision's ToTensor transform assumes that whatever it is given
        is always in channels_last format (as is usually the case with PIL
        images) and always returns images with the channels *first*!
        
            Converts a PIL Image or numpy.ndarray (H x W x C) in the range
            [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range
            [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P,
            I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has
            dtype = np.uint8
        """
        return image_to_tensor(image)

    # @classmethod
    # def shape_change(cls, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    #     from .channels import ChannelsFirstIfNeeded
    #     return ChannelsFirstIfNeeded.shape_change(input_shape)

    # @classmethod
    # def space_change(cls, input_space: gym.Space) -> gym.Space:
    #     if not isinstance(input_space, spaces.Box):
    #         logger.warning(UserWarning(f"Transform {cls} is only meant for Box spaces, not {input_space}"))
    #         return input_space
    #     return spaces.Box(
    #         low=0.,
    #         high=1.,
    #         shape=cls.shape_change(input_space.shape),
    #         dtype=np.float32,
    #     )
        