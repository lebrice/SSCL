""" WIP (@lebrice): Playing around with the idea of using a typed object to
represent the different forms of "batches" that settings produce and that
different models expect.
"""
import dataclasses
import itertools
import operator
from abc import ABC
from collections import abc as collections_abc
from collections import namedtuple
from dataclasses import dataclass
from functools import partial, singledispatch
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    KeysView, List, Mapping, NamedTuple, Optional, Sequence,
                    Set, Tuple, Type, TypeVar, Union)

import gym
import numpy as np
import torch
from gym import spaces
from sequoia.utils.generic_functions import singledispatchmethod
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import zip_dicts
from sequoia.utils.categorical import Categorical
from torch import Tensor

logger = get_logger(__file__)

B = TypeVar("B", bound="Batch", covariant=True)
T = TypeVar("T", Tensor, np.ndarray, "Batch")
V = TypeVar("V")

def hasmethod(obj: Any, method_name: str) -> bool:
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))


@dataclass(frozen=True, eq=False)
class Batch(ABC, Mapping[str, T]):
    """ Abstract base class for typed, immutable objects holding tensors.
    
    Can be used as an immutable dictionary mapping from strings to tensors, or
    as a tuple if you index with an integer.
    Also has some Tensor-like helper methods like `to()`, `numpy()`, `detach()`,
    etc.
    
    Other features:
    - numpy-style indexing/slicing/masking
    - moving all items between devices
    - changing the dtype of all tensors
    - detaching all tensors
    - Convertign all tensors to numpy arrays
    - convertible to a tuple or a dict

    NOTE: Using dataclasses rather than namedtuples, because those aren't really
    meant to be subclassed, so we couldn't use them to make the 'Observations'
    hierarchy, for instance.
    Dataclasses work better for that purpose.

    Examples:

    >>> import torch
    >>> from typing import Optional
    >>> from dataclasses import dataclass
    
    >>> @dataclass(frozen=True)
    ... class MyBatch(Batch):
    ...     x: Tensor
    ...     y: Tensor = None
    
    >>> batch = MyBatch(x=torch.ones([10, 3, 32, 32]), y=torch.arange(10))
    >>> batch.shapes
    {'x': torch.Size([10, 3, 32, 32]), 'y': torch.Size([10])}
    >>> batch.batch_size
    10
    >>> batch.dtypes
    {'x': torch.float32, 'y': torch.int64}
    >>> batch.dtype # No shared dtype, so dtype returns None.
    >>> batch.float().dtype # Converting the all items to float dtype:
    torch.float32
    
    Device-related methods:
    
        
    >>> from dataclasses import dataclass
    >>> import torch
    >>> from torch import Tensor
    
    >>> @dataclass(frozen=True)
    ... class Observations(Batch):
    ...     x: Tensor
    ...     task_labels: Tensor
    ...     done: Tensor
    ...
    >>> # Example: observations from two gym environments (e.g. VectorEnv) 
    >>> observations = Observations(
    ...     x = torch.arange(10).reshape([2, 5]),
    ...     task_labels = torch.arange(2, dtype=int),
    ...     done = torch.zeros(2, dtype=bool),
    ... )
    
    >>> observations.shapes
    {'x': torch.Size([2, 5]), 'task_labels': torch.Size([2]), 'done': torch.Size([2])}
    >>> observations.batch_size
    2
    
    Datatypes:
    
    >>> observations.dtypes
    {'x': torch.int64, 'task_labels': torch.int64, 'done': torch.bool}
    >>> observations.dtype # No shared dtype, so dtype returns None.
    >>> observations.float().dtype # Converting the all items to float dtype:
    torch.float32
    
    
    Returns the device common to all items, or None:
    
    >>> observations.device  
    device(type='cpu')
    >>> # observations.to("cuda").device
    >>> # device(type='cuda', index=0)
    
    >>> observations[0]
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    
    Additionally, when slicing a Batch across the first dimension, you get
    other typed objects as a result! For example:    
    
    >>> observations[:, 0]
    Observations(x=tensor([0, 1, 2, 3, 4]), task_labels=tensor(0), done=tensor(False))
    
    >>> observations[:, 1]
    Observations(x=tensor([5, 6, 7, 8, 9]), task_labels=tensor(1), done=tensor(False))
    """
    # TODO: Would it make sense to add a gym Space class variable here? 
    space: ClassVar[Optional[gym.Space]]
    # TODO: Remove these:
    field_names: ClassVar[List[str]]
    _namedtuple: ClassVar[Type[NamedTuple]]

    def __init_subclass__(cls, *args, **kwargs):
        # IDEA: By not marking 'Batch' a dataclass, we would let the subclass
        # decide it if wants to be frozen or not!
        
        # Subclasses of `Batch` should be dataclasses!
        if not dataclasses.is_dataclass(cls):
            raise RuntimeError(f"{__class__} subclass {cls} must be a dataclass!")
        super().__init_subclass__(*args, **kwargs)

    def __post_init__(self):
        # Create some class attributes, if they don't already exist.
        # TODO: We have to set these here because __init_subclass__ is called
        # before the dataclasses package sets the 'fields' attribute, it seems.
        cls = type(self)
        if "field_names" not in cls.__dict__:
            type(self).field_names = [f.name for f in dataclasses.fields(self)]
        # Create a NamedTuple type for this new subclass.
        if "_named_tuple" not in cls.__dict__:
            type(self)._namedtuple = namedtuple(type(self).__name__ + "Tuple", self.field_names)

    def __iter__(self) -> Iterator[str]:
        """ Yield the 'keys' of this object, i.e. the names of the fields. """
        return iter(self.field_names)

    def __len__(self) -> int:
        """ Returns the number of fields. """
        return len(self.field_names)
    
    def __eq__(self, other: Union["Batch", Any]) -> bool:
        # Not sure this is useful.
        return NotImplemented
    
        if not isinstance(other, Batch):
            return NotImplemented
        if type(self) != type(other):
            # Not allowing these sorts of comparisons.
            return NotImplemented
        items_equal = {
            k: v == other[k]
            for k, v in self.items()
        }
        return all(
            is_equal.all() if isinstance(is_equal, (Tensor, np.ndarray)) else is_equal
            for is_equal in items_equal.values()
        )
        
        
    @singledispatchmethod
    def __getitem__(self, index: Any) -> T:
        """ Select a subset of the fields of this object. Can also be indexed
        with tuples, boolean numpy arrays or tensors, as well as None. 
        """
        raise KeyError(index)
    
    @__getitem__.register(type(None))
    def _getitem_none(self, index: None) -> "Batch":
        """ Indexing with 'None' gives back a copy with all the items having an
        extra batch dimension.
        """
        return self.with_batch_dimension()
        return getattr(self, index)

    @__getitem__.register
    def _getitem_by_name(self, index: str) -> Union[Tensor, Any]:
        return getattr(self, index)

    @__getitem__.register
    def _getitem_by_index(self, index: int) -> Union[Tensor, Any]:
        return getattr(self, self.field_names[index])

    @__getitem__.register(slice)
    def _getitem_with_slice(self, index: slice) -> "Batch":
        # NOTE: I don't think it would be a good idea to support slice indexing,
        # as it could be confusing and give the user the impression that it
        # is slicing into the tensors, rather than into the fields.
        # I guess this might be doable, but is it really useful?
        raise NotImplementedError(
            "Batch objects don't support indexing with (just) slices atm."
        )
        if index == slice(None, None, None) or index == slice(0, len(self), 1):
            return self

    @__getitem__.register(type(Ellipsis))
    def _(self: B, index) -> B:
        return self

    @__getitem__.register(np.ndarray)
    @__getitem__.register(Tensor)
    def _getitem_with_array(self, index: np.ndarray) -> B:
        """
        NOTE: Indexing with just an array uses the array as a 'mask' on all
        fields, instead of indexing the "keys" of this object.
        """
        assert len(index) == self.batch_size
        return self[:, index]
    
    @__getitem__.register(tuple)
    def _getitem_with_tuple(self, index: Tuple[Union[slice, Tensor, np.ndarray, int], ...]):
        """ When slicing with a tuple, if the first item is an integer, we get
        the attribute at that index and slice it with the rest.
        For now, the first item in the tuple can only be either an int or an
        empty slice.
        """
        if len(index) <= 1:
            raise IndexError(f"Invalid index {index}: When indexing with "
                             f"tuples or lists, they need to have len > 1.")
        field_index = index[0]
        item_index = index[1:]
        # if len(item_index) == 1:
        #     item_index = item_index[0]

        if isinstance(field_index, int):
            # logger.debug(f"Getting the {field_index}'th field, with slice {index[1:]}")
            return self[field_index][item_index]

        # e.g: forward_pass[:, 1]
        if isinstance(field_index, slice):
            if field_index == slice(None):
                # logger.debug(f"Indexing all fields {field_index} with index: {item_index}")
                return type(self)(**{
                    key: (
                        value[index] if isinstance(value, Batch) else
                        value[item_index] if value is not None else None
                    )
                    for key, value in self.items()
                })

        # batch[..., 0] : Not sure this would really be that helpful.
        if field_index == Ellipsis:
            logger.debug(f"Using ellipsis (...) as the field index?")
            return type(self)(**{
                key: value[Ellipsis, item_index] if value is not None else None 
                for key, value in self.items()
            })
        
        raise NotImplementedError(
            f"Only support tuple indexing with emptyslices or int as first "
            f"tuple item for now. (index={index})"
        )

    def slice(self: B, index: Union[int, slice, np.ndarray, Tensor]) -> B:
        """ Gets a slice across the first (batch) dimension.
        Raises an error if there is no batch size.
        
        Always returns an object with a batch dimension, even when `index` has len of 1.
        """
        if not isinstance(index, (int, slice, np.ndarray, Tensor)):
            raise NotImplementedError(f"can't slice with index {index}")

        # BUG: By putting a 'None' value in the ForwardPass
        def getitem_if_val_is_not_none(val, index):
            if val is None:
                return None
            return val[index]

        sliced_value = self._map(partial(getitem_if_val_is_not_none, index=index), recursive=True)
        if isinstance(index, int):
            sliced_value = sliced_value.with_batch_dimension()
        return sliced_value
        # return type(self)(**{
        #     k: v.slice(index) if isinstance(v, Batch) else
        #     v[index] if v is not None else None
        #     for k, v in self.items()
        # })

    def __setitem__(self, index: Union[int, str], value: Any):
        """ Set a value in slices of one or more of the fields.

        NOTE: Since this class is marked as frozen, we can't change the
        attributes, so the index should be a tuple (to change parts of the
        tensors, for instance.
        """
        if not isinstance(index, tuple) or len(index) < 2:
            raise NotImplementedError("index needs to be tuple with len >= 2")
        # Get which keys/fields were selected:
        selected_fields = np.array(self.field_names)[index[0]]
        for selected_field in selected_fields:
            item = self[selected_field]
            if item is not None:
                item[index[1:]] = value

    def keys(self) -> KeysView[str]:
        return KeysView(self.field_names)

    def values(self) -> Tuple[T, ...]:
        return self.as_namedtuple()

    def items(self) -> Iterable[Tuple[str, T]]:
        for name in self.field_names:
            yield name, getattr(self, name)

    @property
    def devices(self) -> Dict[str, Union[Optional[torch.device], Dict]]:
        """ Dict from field names to their device if they have one, else None.
        
        If `self` has `Batch` fields, the values for those will be dicts.
        """
        return {
            k: v.devices if isinstance(v, Batch) else getattr(v, "device", None)
            for k, v in self.items()
        }

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the device common to all items, or `None`.

        Returns
        -------
        Tuple[Optional[torch.device]]
            None if the devices are unknown/different, or the common device.
        """
        device: Optional[torch.device] = None
        # TODO: These kinds of methods can't discriminate between a child item
        # having all all None tensors and it having different devices atm.
        for key, value in self.items():
            if isinstance(value, Batch):
                item_device = value.device
                if item_device is None:
                    # Child item doesn't have a 'device', so `self` also doesnt.
                    return None
            else:
                item_device = getattr(value, "device", None)
            
            if item_device is None:
                continue
            if device is None:
                device = item_device
            elif item_device != device:
                return None
        return device

    @property
    def dtypes(self) -> Dict[str, Union[Optional[torch.dtype], Dict]]:
        """ Dict from field names to their dtypes if they have one, else None.
        
        If `self` has `Batch` fields, the values for those will be dicts.
        """
        return {
            k: v.dtypes if isinstance(v, Batch) else getattr(v, "dtype", None)
            for k, v in self.items()
        }

    @property
    def dtype(self) -> Tuple[Optional[torch.dtype]]:
        """Returns the dtype common to all tensors, or None.

        Returns
        -------
        Dict[Optional[torch.dtype]]
            The common dtype, or `None` if the dtypes are unknown/different.
        """
        dtype: Optional[torch.dtype] = None
        
        for key, value in self.items():
            item_dtype = getattr(value, "dtype", None)
            if item_dtype is None:
                continue
            if dtype is None:
                dtype = item_dtype
            elif item_dtype != dtype:
                return None
        return dtype

    def as_namedtuple(self) -> Tuple[T, ...]:
        return self._namedtuple(**{
            k: v for k, v in self.items()
        })
    
    def as_list_of_tuples(self) -> Iterable[Tuple[T, ...]]:
        """Returns an iterable of the items in the 'batch', each item as a
        namedtuple (list of tuples).
        """
        # If one of the fields is None, then we convert it into a list of Nones,
        # so we can zip all the fields to create a list of tuples.
        field_items = [
            [items for _ in range(self.batch_size)] if items is None or items is {} else
            [item for item in items]
            for items in self.as_tuple()
        ]
        assert all([len(items) == self.batch_size for items in field_items])
        return list(itertools.starmap(self._namedtuple, zip(*field_items)))

    def as_tuple(self) -> Tuple[T, ...]:
        """Returns a namedtuple containing the 'batched' attributes of this
        object (tuple of lists).
        """
        # TODO: Turning on the namedtuple return value by default.
        # return tuple(
        #     getattr(self, f.name) for f in dataclasses.fields(self)
        # )
        return self.as_namedtuple()

    # def as_dict(self) -> Dict[str, T]:
    #     # NOTE: dicts are ordered since python 3.7
    #     return {
    #         field_name: getattr(self, field_name)
    #         for field_name in self.field_names
    #     }

    def to(self, *args, **kwargs):
        def _to(item, *args_, **kwargs_):
            if hasattr(item, "to") and callable(item.to):
                return item.to(*args_, **kwargs_)
            return item
        return self._map(_to, *args, **kwargs, recursive=True)

    def float(self, dtype=torch.float):
        return self.to(dtype=dtype)
    
    def float32(self, dtype=torch.float32):
        return self.to(dtype=dtype)

    def int(self, dtype=torch.int):
        return self.to(dtype=dtype)

    def double(self, dtype=torch.double):
        return self.to(dtype=dtype)

    def numpy(self):
        """Returns a new Batch object of the same type, with all Tensors
        converted to numpy arrays.

        Returns
        -------
        [type]
            [description]
        """
        def _numpy(v):
            if isinstance(v, (Tensor, Batch)):
                return v.detach().cpu().numpy()
            return v
        return self._map(_numpy, recursive=True)
        # return type(self)(**{
        #     k: v.detach().cpu().numpy() if isinstance(v, (Tensor, Batch)) else v
        #     for k, v in self.items()
        # })

    def detach(self):
        """Returns a new Batch object of the same type, with all Tensors
        detached.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors detached.
        """
        from sequoia.utils.generic_functions import detach
        return self._map(detach)
        # return type(self)(**detach({
        #     k: v.detach() if isinstance(v, (Tensor, Batch)) else v for k, v in self.items()
        # }))

    def cpu(self, **kwargs):
        """Returns a new Batch object of the same type, with all Tensors
        moved to cpu.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors moved to CPU.
        """
        return self.to(device="cpu", **kwargs)

    def cuda(self, device=None, **kwargs):
        """Returns a new Batch object of the same type, with all Tensors
        moved to cuda device.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors moved to cuda.
        """
        return self.to(device=(device or "cuda"), **kwargs)

    @property
    def shapes(self) -> Dict[str, Union[torch.Size, Dict]]:
        """ Dict from field names to their shapes if they have one, else None.
        
        If `self` has `Batch` fields, the values for those will be dicts.
        """
        return {
            k: v.shapes if isinstance(v, Batch) else getattr(v, "shape", None)
            for k, v in self.items()
        }

    @property
    def batch_size(self) -> Optional[int]:
        """ Returns the length of the first dimension if it is common to all
        tensors in this object, else None.
        """
        # NOTE: If all tensors have just one dimension and are all the same
        # length, then this would give back that length.
        batch_size: Optional[int] = None
        for k, v in self.items():
            if isinstance(v, Batch):
                v_batch_size = v.batch_size
                if v_batch_size is None:
                    # child item doesn't have a batch size, so we dont either.
                    return None
                elif batch_size is None:
                    batch_size = v_batch_size
                elif v_batch_size != batch_size:
                    return None
            else:
                item_shape = getattr(v, "shape", None)
                if item_shape is None:
                    continue
                if not item_shape:
                    return None
                v_batch_size = item_shape[0] 
                if batch_size is None:
                    batch_size = v_batch_size
                elif v_batch_size != batch_size:
                    return None
        return batch_size

    def with_batch_dimension(self: B) -> B:
        """ Returns a copy of `self` where all numpy arrays / tensors have an
        extra `batch` dimension of size 1.
        """
        # TODO: Do we 'wrap' the `None` values? or keep them as-is?
        @singledispatch
        def unsqueeze(v: Any) -> Any:
            if v is None:
                return v
            return np.asarray([v])

        @unsqueeze.register(Categorical)
        @unsqueeze.register(np.ndarray)
        @unsqueeze.register(Tensor)
        def _unsqueeze_array(v: Union[np.ndarray, Tensor, Categorical]) -> Union[np.ndarray, Tensor, Categorical]:
            return v[None]

        return self._map(unsqueeze)

    def remove_batch_dimension(self: B) -> B:
        """ Returns a copy of `self` where all numpy arrays / tensors have an
        the extra `batch` dimension removed.

        Raises an error if any non-None value doesn't have a batch dimension of
        size 1. 
        """
        return self[:, 0]

    def split(self: B) -> List[B]:
        """Returns an iterable of the items in the 'batch', each item as a
        object of the same type as `self`.
        """
        # If one of the fields is None, then we convert it into a list of Nones,
        # so we can zip all the fields to create a list of tuples.
        return [self[:, i] for i in range(self.batch_size)]

    @classmethod
    def stack(cls: Type[B], items: List[B]) -> B:
        items = list(items)
        from sequoia.utils.generic_functions import stack
        # Just to make sure that the returned item will be of the type `cls`.
        assert isinstance(items[0], cls)
        return stack(items)

    @classmethod
    def concatenate(cls: Type[B], items: List[B], **kwargs) -> B:
        items = list(items)
        from sequoia.utils.generic_functions import concatenate
        assert isinstance(items[0], cls)
        return concatenate(items, **kwargs)
    
    def torch(self, device: Union[str, torch.device] = None, dtype: torch.dtype = None):
        """ Converts any ndarrays to Tensors if possible and returns a new
        object of the same type.
        
        NOTE: This is the opposite of `self.numpy()`
        """
        def _from_numpy(v: Union[np.ndarray, Any]) -> Union[Tensor, Any]:
            try:
                return torch.as_tensor(v, device=device, dtype=dtype)
            except (TypeError, RuntimeError):
                return v
        return self._map(_from_numpy, recursive=True)
    
    def _map(self: B,
             func: Callable,
             *args,
             recursive: bool = True,
             **kwargs) -> B:
        """ Returns an object of the same type as `self`, where function `func`
        has been applied (with positional args `args` and keyword-arguments
        `kwargs`) to all its values, (inluding the values of nested `Batch`
        objects if `recursive` is True). 
        """
        new_items = {}
        for key, value in self.items():
            if isinstance(value, Batch):
                if not recursive:
                    # don't apply the function to nested Batch objects unless
                    # `recursive` is True.
                    new_items[key] = value
                else:
                    new_items[key] = value._map(func, *args, recursive=recursive, **kwargs)
            else:
                new_items[key] = func(value, *args, **kwargs)  # type: ignore
        return type(self)(**new_items)

    def _apply(self: B,
               func: Callable[[T, Any], None],
               *args,
               recursive: bool = True,
               **kwargs) -> None:
        """ Applies function `func` to all the values in `self`, and optionally
        to all its nested values when `recursive` is True. 
        
        Returns None, as this assumes that `func` modifies the values in-place.
        """
        for key, value in self.items():
            if isinstance(value, Batch) and not recursive:
                # Skip any Batch objects if `recursive` is False.
                continue
            func(value, *args, **kwargs)  # type: ignore


if __name__ == "__main__":
    import doctest
    doctest.testmod()
