""" Generic function for concatenating ndarrays/tensors/distributions/Mappings
etc.
"""
from collections.abc import Mapping
from functools import singledispatch
from typing import Any, Dict, List, Sequence, Union, TypeVar, overload

import numpy as np
import torch
from gym import Space, spaces
from sequoia.common.spaces.sparse import Sparse
from sequoia.utils.categorical import Categorical
from torch import Tensor

T = TypeVar("T")


# @overload
# def stack(first_item: List[T]) -> Sequence[T]:
#     ...

# @overload
# def stack(first_item: T, *others: T) -> Sequence[T]:
#     ...


@singledispatch
def stack(first_item: Union[T, List[T]], *others: T, **kwargs) -> Any:
    # By default, if we don't know how to handle the item type, just
    # return an ndarray with with all the items.
    # note: We could also try to return a tensor, rather than an ndarray
    # but I'd rather keep it simple for now.
    if not others:
        # If this was called like stack(tensor_list), then we just split off
        # the list of items.
        if first_item is None:
            # Stacking a list of 'None' items returns None.
            return None
        assert isinstance(first_item, (list, tuple)), first_item
        # assert len(first_item) > 1, first_item
        items = first_item
        return stack(items[0], *items[1:], **kwargs)
    np_stack_kwargs = kwargs.copy()
    if "dim" in np_stack_kwargs:
        np_stack_kwargs["axis"] = np_stack_kwargs.pop("dim")
    return np.stack([first_item, *others], **np_stack_kwargs)


@stack.register(type(None))
def _stack_none(
    first_item: None, *others: None, **kwargs
) -> Union[None, np.ndarray]:
    # TODO: Should we return an ndarray with 'None' entries, of dtype np.object_? or 
    # just a single None?
    # Opting for a single None for now, as it's easier to work with. (`v is None` works)
    if all(v is None for v in others):
        return None
    return np.array([first_item, *others])
    # if not others:
    #     return None
    # return np.array([None, *others])


@stack.register(np.ndarray)
def _stack_ndarrays(
    first_item: np.ndarray, *others: np.ndarray, **kwargs
) -> np.ndarray:
    return np.stack([first_item, *others], **kwargs)


@stack.register(Tensor)
def _stack_tensors(first_item: Tensor, *others: Tensor, **kwargs) -> Tensor:
    return torch.stack([first_item, *others], **kwargs)


@stack.register(Mapping)
def _stack_dicts(first_item: Dict, *others: Dict, **kwargs) -> Dict:
    return type(first_item)(
        **{
            key: stack(first_item[key], *(other[key] for other in others), **kwargs)
            for key in first_item.keys()
        }
    )


@stack.register(Categorical)
def _stack_distributions(
    first_item: Categorical, *others: Categorical, **kwargs
) -> Categorical:
    return Categorical(
        logits=torch.stack(
            [first_item.logits, *(other.logits for other in others)], **kwargs
        )
    )
