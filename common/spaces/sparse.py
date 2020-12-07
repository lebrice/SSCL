""" 'wrapper' around a gym.Space that adds has a probability of sampling `None`
instead of a sample from the 'base' space.

As a result, `None` is always a valid sample from any Sparse space.


TODO: Totally optional, but if we wanted to use the `shared_memory=True`
argument to the AsyncVectorEnv or BatchedVectorEnv wrappers, we'd need to
test/debug some bugs with shared memory functions below. In the interest of time
though, I just set that `shared_memory=False`, and it works great.  
"""
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gym
import numpy as np
from gym import spaces

T = TypeVar("T")


class Sparse(gym.Space, Generic[T]):
    
    # TODO: Rename 'none_prob' to 'sparsity', because it sounds fancier.
    def __init__(self, base: gym.Space, none_prob: float = 0.):
        self.base = base
        self.none_prob = none_prob
        # Would it ever cause a problem to have different dtypes for different
        # instances of the same space?
        # dtype = self.base.dtype if none_prob != 0. else np.object_ 
        super().__init__(shape=self.base.shape, dtype=np.object_)

    def seed(self, seed=None):
        super().seed(seed)
        return self.base.seed(seed=seed)

    def sample(self) -> Optional[T]:
        if self.none_prob == 0:
            return self.base.sample()
        if self.none_prob == 1.:
            return None
        p = self.np_random.random()
        if p <= self.none_prob:
            return None
        else:
            return self.base.sample()

    def contains(self, x: Optional[T]):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return x is None or self.base.contains(x)

    def __repr__(self):
        return f"Sparse({self.base}, none_prob={self.none_prob})"
    
    def __eq__(self, other: Any):
        if not isinstance(other, Sparse):
            return NotImplemented
        return other.base == self.base and other.none_prob == self.none_prob


    def to_jsonable(self, sample_n):
        assert False, sample_n
        super().to_jsonable
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        assert False, sample_n
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

# from gym.spaces.utils import flatdim, flatten
from functools import singledispatch

import gym.spaces.utils
import gym.vector.utils
from gym.vector.utils import (batch_space, concatenate, create_empty_array,
                              create_shared_memory)
import gym.vector.utils.numpy_utils
# Customize how these functions handle `Sparse` spaces by making them
# singledispatch callables and registering a new callable.

def _is_singledispatch(module_function):
    return hasattr(module_function, "registry")


def register_sparse_variant(module, module_fn_name: str):
    """ Converts a function from the given module to a singledispatch callable,
    and registers the wrapped function as the callable to use for Sparse spaces.
    
    The module function must have the space as the first argument for this to
    work.
    """
    module_function = getattr(module, module_fn_name)
    
    # Convert the function to a singledispatch callable.
    if not _is_singledispatch(module_function):
        module_function = singledispatch(module_function)
        setattr(module, module_fn_name, module_function)
    # Register the function as the callable to use when the first arg is a
    # Sparse object.
    def wrapper(function):
        module_function.register(Sparse, function)
        return function
    return wrapper


@register_sparse_variant(gym.spaces.utils, "flatdim")
def flatdim_sparse(space: Sparse) -> int:
    return gym.spaces.utils.flatdim(space.base)

@register_sparse_variant(gym.spaces.utils, "flatten")
def flatten_sparse(space: Sparse[T], x: Optional[T]) -> Optional[np.ndarray]:
    return np.array([None]) if x is None else gym.spaces.utils.flatten(space.base, x)

@register_sparse_variant(gym.spaces.utils, "flatten_space")
def flatten_sparse_space(space: Sparse[T]) -> Optional[np.ndarray]:
    space = gym.spaces.utils.flatten_space(space.base)
    space.dtype = np.object_
    return space
    
@register_sparse_variant(gym.spaces.utils, "unflatten")
def unflatten_sparse(space: Sparse[T], x: np.ndarray) -> Optional[T]:
    if len(x) == 1 and x[0] is None:
        return None
    else:
        return gym.spaces.utils.unflatten(space.base, x)


@register_sparse_variant(gym.vector.utils.numpy_utils, "create_empty_array")
def create_empty_array_sparse(space: Sparse, n=1, fn=np.zeros) -> np.ndarray:
    return fn([n], dtype=np.object_)

import multiprocessing as mp
from multiprocessing import Array, Value
from multiprocessing.context import BaseContext

import gym.vector.utils.shared_memory

from gym.vector.utils.shared_memory import write_base_to_shared_memory
from ctypes import c_bool


@register_sparse_variant(gym.vector.utils.shared_memory, "create_shared_memory")
def create_shared_memory_for_sparse_space(space: Sparse, n: int = 1, ctx: BaseContext = mp):
    # The shared memory should be something that can accomodate either 'None'
    # or a sample from the space. Therefore we should probably just create the
    # array for the base space, but then how would store a 'None' value in that
    # space?
    # What if we return a tuple or something, in which we actually add an 'is-none'
    print(f"Creating shared memory for {n} entries from space {space}")
    
    return {
        "is_none": ctx.Array(c_bool, np.zeros(n, dtype=np.bool)),
        "value": gym.vector.utils.shared_memory.create_shared_memory(space.base, n, ctx)
    }


@register_sparse_variant(gym.vector.utils.shared_memory, "write_to_shared_memory")
def write_to_shared_memory(index: int,
                           value: Optional[T],
                           shared_memory: Union[Dict, Tuple, BaseContext.Array],
                           space: Union[Sparse[T], gym.Space]):
    print(f"Writing entry from space {space} at index {index} in shared memory")
    if isinstance(space, Sparse):
        assert isinstance(shared_memory, dict)
        is_none_array = shared_memory["is_none"]
        value_array = shared_memory["value"]
        assert False, index
        assert False, is_none_array

        is_none_array[index] = value is None

        if value is not None:
            return write_to_shared_memory(index, value, value_array, space.base)
    else:
        # TODO: Would this cause a problem, say in the case where we have a
        # regular space like Tuple that contains some Sparse spaces, then would
        # calling this "old" function here prevent this "new" function from
        # being used on the children?
        return gym.vector.utils.shared_memory(index, value, shared_memory, space)


from gym.vector.utils.shared_memory import \
    read_from_shared_memory as read_from_shared_memory_


@register_sparse_variant(gym.vector.utils.shared_memory, "read_from_shared_memory")
def read_from_shared_memory(shared_memory: Union[Dict, Tuple, BaseContext.Array],
                            space: Sparse,
                            n: int = 1):
    print(f"Reading {n} entries from space {space} from shared memory")
    if isinstance(space, Sparse):
        assert isinstance(shared_memory, dict)
        is_none_array = list(shared_memory["is_none"])
        value_array = shared_memory["value"]
        assert len(is_none_array) == len(value_array) == n
        
        # This might include some garbage (or default) values, which weren't
        # set.
        read_values = read_from_shared_memory(value_array, space.base, n)
        print(f"Read values from space: {read_values}")
        print(f"is_none array: {list(is_none_array)}")
        # assert False, (list(is_none_array), read_values, space)
        values = [
            None if is_none_array[index] else read_values[index]
            for index in range(n)
        ]
        print(f"resulting values: {values}")
        return values
        return read_from_shared_memory_(shared_memory, space.base, n)
    return read_from_shared_memory_(shared_memory, space, n)

    

@register_sparse_variant(gym.vector.utils, "batch_space")
def batch_sparse_space(space: Sparse, n: int=1) -> gym.Space:
    # NOTE: This means we do something different depending on the sparsity.
    # Could that become an issue?
    assert _is_singledispatch(batch_space)
    
    sparsity = space.none_prob
    if sparsity == 0.:
        # If the space has 0 sparsity, then batch it just like you would its
        # base space.
        # TODO: This is convenient, but not very consistent, as the length of
        # the batches changes depending on the sparsity of the space..
        return Sparse(batch_space(space.base, n), none_prob=0.)

    # Sticking to the default behaviour from gym for now, which is to just
    # return a tuple of length n with n copies of the space.
    return spaces.Tuple(tuple(space for _ in range(n)))

    # We could also do this, where we make the sub-spaces sparse:
    # batch_space(Sparse<Tuple<A, B>>) -> Tuple<batch_space(Sparse<A>), batch_space(Sparse<B>)>

    if isinstance(space.base, spaces.Tuple):
        return spaces.Tuple([
            spaces.Tuple([Sparse(sub_space, sparsity) for _ in range(n)])
            for sub_space in space.base.spaces
        ])
    if isinstance(space.base, spaces.Dict):
        return spaces.Dict({
            name: Sparse(batch_space(sub_space, n), sparsity)
            for name, sub_space in space.base.spaces.items()
        })

    return batch_space(space.base, n)


@register_sparse_variant(gym.vector.utils, "concatenate")
def concatenate_sparse_spaces(space: Sparse, n: int=1) -> gym.Space:
    assert False, f"Debugging: {space}, {n}"
    return concatenate(space.base, n)
