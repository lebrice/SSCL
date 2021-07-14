""" Miscelaneous utility functions. """
import collections
import functools
import hashlib
import inspect
import itertools
import operator
import os
import random
import re
import warnings
from collections import defaultdict, deque
from collections.abc import MutableMapping
from dataclasses import Field, fields
from functools import reduce
from inspect import getsourcefile, isabstract, isclass
from itertools import filterfalse, groupby
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
from simple_parsing import field
from torch import Tensor, cuda, nn

cuda_available = cuda.is_available()
gpus_available = cuda.device_count()

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

Dataclass = TypeVar("Dataclass")


def field_dict(dataclass: Dataclass) -> Dict[str, Field]:
    return {field.name: field for field in fields(dataclass)}


def mean(values: Iterable[T]) -> T:
    values = list(values)
    return sum(values) / len(values)


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def n_consecutive(
    items: Iterable[T], n: int = 2, yield_last_batch=True
) -> Iterable[Tuple[T, ...]]:
    """Collect data into chunks of up to `n` elements.
    
    When `yield_last_batch` is True, the final chunk (which might have fewer
    than `n` items) will also be yielded.
    
    >>> list(n_consecutive("ABCDEFG", 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    values: List[T] = []
    for item in items:
        values.append(item)
        if len(values) == n:
            yield tuple(values)
            values.clear()
    if values and yield_last_batch:
        yield tuple(values)


def fix_channels(x_batch: Tensor) -> Tensor:
    # TODO: Move this to data_utils.py
    if x_batch.dim() == 3:
        return x_batch.unsqueeze(1)
    else:
        if x_batch.shape[1] != min(x_batch.shape[1:]):
            return x_batch.transpose(1, -1)
        else:
            return x_batch


def to_dict_of_lists(
    list_of_dicts: Iterable[Dict[str, Any]]
) -> Dict[str, List[Tensor]]:
    """ Returns a dict of lists given a list of dicts.
    
    Assumes that all dictionaries have the same keys as the first dictionary.
    
    Args:
        list_of_dicts (Iterable[Dict[str, Any]]): An iterable of dicts.
    
    Returns:
        Dict[str, List[Tensor]]: A Dict of lists.
    """
    result: Dict[str, List[Any]] = defaultdict(list)
    for i, d in enumerate(list_of_dicts):
        for key, value in d.items():
            result[key].append(value)
        assert (
            d.keys() == result.keys()
        ), f"Dict {d} at index {i} does not contain all the keys!"
    return result


def add_prefix(some_dict: Dict[str, T], prefix: str = "", sep=" ") -> Dict[str, T]:
    """Adds the given prefix to all the keys in the dictionary that don't already start with it. 
    
    Parameters
    ----------
    - some_dict : Dict[str, T]
    
        Some dictionary.
    - prefix : str, optional, by default ""
    
        A string prefix to append.
    
    - sep : str, optional, by default " "

        A string separator to add between the `prefix` and the existing keys
        (which do no start by `prefix`). 

    
    Returns
    -------
    Dict[str, T]
        A new dictionary where all keys start with the prefix.


    Examples:
    -------
    >>> add_prefix({"a": 1}, prefix="bob", sep="")
    {'boba': 1}
    >>> add_prefix({"a": 1}, prefix="bob")
    {'bob a': 1}
    >>> add_prefix({"a": 1}, prefix="a")
    {'a': 1}
    >>> add_prefix({"a": 1}, prefix="a ")
    {'a': 1}
    >>> add_prefix({"a": 1}, prefix="a", sep="/")
    {'a': 1}
    """
    if not prefix:
        return some_dict
    result: Dict[str, T] = type(some_dict)()

    if sep and prefix.endswith(sep):
        prefix = prefix.rstrip(sep)

    for key, value in some_dict.items():
        new_key = key if key.startswith(prefix) else (prefix + sep + key)
        result[new_key] = value
    return result


def loss_str(loss_tensor: Tensor) -> str:
    loss = loss_tensor.item()
    if loss == 0:
        return "0"
    elif abs(loss) < 1e-3 or abs(loss) > 1e3:
        return f"{loss:.1e}"
    else:
        return f"{loss:.3f}"


def set_seed(seed: int):
    """ Set the pytorch/numpy random seed. """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_identity(size: int = 16, **sample) -> str:
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    size: int
        size of the unique hash

    **sample:
        Dictionary to compute the hash from

    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode("utf8"))

        if isinstance(v, dict):
            sample_hash.update(compute_identity(size, **v).encode("utf8"))
        else:
            sample_hash.update(str(v).encode("utf8"))

    return sample_hash.hexdigest()[:size]


def prod(iterable: Iterable[T]) -> T:
    """ Like sum() but returns the product of all numbers in the iterable.

    >>> prod(range(1, 5))
    24
    """
    return reduce(operator.mul, iterable, 1)


def common_fields(a, b) -> Iterable[Tuple[str, Tuple[Field, Field]]]:
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    a_fields = fields(a)
    b_fields = fields(b)
    for field_a in a_fields:
        name_a: str = field_a.name
        value_a = getattr(a, field_a.name)
        for field_b in b_fields:
            name_b: str = field_b.name
            value_b = getattr(b, field_b.name)
            if name_a == name_b:
                yield name_a, (value_a, value_b)


def add_dicts(d1: Dict, d2: Dict, add_values=True) -> Dict:
    result = d1.copy()
    for key, v2 in d2.items():
        if key not in d1:
            result[key] = v2
        elif isinstance(v2, dict):
            result[key] = add_dicts(d1[key], v2, add_values=add_values)
        elif not add_values:
            result[key] = v2
        else:
            result[key] = d1[key] + v2
    return result


def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj: Any, attr: str, *args):
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and len(list(path.iterdir())) > 0


D = TypeVar("D", bound=Dict)


def flatten_dict(d: D, separator: str = "/") -> D:
    """Flattens the given nested dict, adding `separator` between keys at different nesting levels.

    Args:
        d (Dict): A nested dictionary
        separator (str, optional): Separator to use. Defaults to "/".

    Returns:
        Dict: A flattened dictionary.
    """
    result = type(d)()
    for k, v in d.items():
        if isinstance(v, dict):
            for ki, vi in flatten_dict(v, separator=separator).items():
                key = f"{k}{separator}{ki}"
                result[key] = vi
        else:
            result[k] = v
    return result


def unique_consecutive(
    iterable: Iterable[T], key: Callable[[T], Any] = None
) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen.

    NOTE: If `key` is passed, it is only used to test for equality, the outputs of `key`
    for each sample won't be returned.

    >>> list(unique_consecutive('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_consecutive('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']

    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def unique_consecutive_with_index(
    iterable: Iterable[T], key: Callable[[T], Any] = None
) -> Iterable[Tuple[int, T]]:
    """List unique elements, preserving order. Remember only the element just seen.
    Yields tuples of the index and the values.

    NOTE: If `key` is passed, it is only used to test for equality, the outputs of `key`
    for each sample won't be returned. If you want to save some compute, use a map as
    the input.

    >>> list(unique_consecutive_with_index('AAAABBBCCDAABBB'))
    [('A', 0), ('B', 4), ('C', 7), ('D', 9), ('A', 10), ('B', 12)]
    >>> list(unique_consecutive_with_index('ABBCcAD', str.lower))
    [('A', 0), ('B', 1), ('C', 3), ('A', 5), ('D', 6)]
    """

    _key = lambda i_v: key(i_v[1]) if key is not None else i_v[1]
    for v, group_iterator in groupby(enumerate(iterable), _key):
        index, first_val = next(group_iterator)
        yield index, first_val


def roundrobin(*iterables: Iterable[T]) -> Iterable[T]:
    """
    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def take(iterable: Iterable[T], n: Optional[int]) -> Iterable[T]:
    """ Takes only the first `n` elements from `iterable`.
    
    if `n` is None, returns the entire iterable.
    """
    return itertools.islice(iterable, n) if n is not None else iterable


def camel_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


def constant(v: T, **kwargs) -> T:
    metadata = kwargs.setdefault("metadata", {})
    metadata["constant"] = v
    metadata["decoding_fn"] = lambda _: v
    metadata["to_dict"] = lambda _: v
    return field(default=v, init=False, **kwargs)


def flag(default: bool, *args, **kwargs):
    return field(default=default, nargs=1, *args, **kwargs)


def dict_union(
    *dicts: Dict[K, V], recurse: bool = True, dict_factory=dict
) -> Dict[K, V]:
    """ Simple dict union until we use python 3.9
    
    If `recurse` is True, also does the union of nested dictionaries.
    NOTE: The returned dictionary has keys sorted alphabetically.

    >>> a = dict(a=1, b=2, c=3)
    >>> b = dict(c=5, d=6, e=7)
    >>> dict_union(a, b)
    {'a': 1, 'b': 2, 'c': 5, 'd': 6, 'e': 7}
    >>> a = dict(a=1, b=dict(c=2, d=3))
    >>> b = dict(a=2, b=dict(c=3, e=6))
    >>> dict_union(a, b)
    {'a': 2, 'b': {'c': 3, 'd': 3, 'e': 6}}
    """
    result: Dict = dict_factory()
    if not dicts:
        return result
    assert len(dicts) >= 1
    all_keys: Set[str] = set()
    all_keys.update(*dicts)
    all_keys = sorted(all_keys)

    # Create a neat generator of generators, to save some memory.
    all_values: Iterable[Tuple[V, Iterable[K]]] = (
        (k, (d[k] for d in dicts if k in d)) for k in all_keys
    )
    for k, values in all_values:
        sub_dicts: List[Dict] = []
        new_value: V = None
        n_values = 0
        for v in values:
            if isinstance(v, dict) and recurse:
                sub_dicts.append(v)
            else:
                # Overwrite the new value for that key.
                new_value = v
            n_values += 1

        if len(sub_dicts) == n_values and recurse:
            # We only get here if all values for key `k` were dictionaries,
            # and if recurse was True.
            new_value = dict_union(*sub_dicts, recurse=True, dict_factory=dict_factory)

        result[k] = new_value
    return result


K = TypeVar("K")
V = TypeVar("V")
M = TypeVar("M")


def zip_dicts(
    *dicts: Dict[K, V], missing: M = None
) -> Iterable[Tuple[K, Tuple[Union[M, V], ...]]]:
    """Iterator over the union of all keys, giving the value from each dict if
    present, else `missing`.
    """
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key, missing) for d in dicts))


def dict_intersection(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[V, ...]]]:
    """Gives back an iterator over the keys and values common to all dicts. """
    dicts = [dict(d.items()) for d in dicts]
    common_keys = set(dicts[0])
    for d in dicts:
        common_keys.intersection_update(d)
    for key in common_keys:
        yield (key, tuple(d[key] for d in dicts))


def try_get(d: Dict[K, V], *keys: K, default: V = None) -> Optional[V]:
    for k in keys:
        try:
            return d[k]
        except KeyError:
            pass
    return default


def remove_suffix(s: str, suffix: str) -> str:
    """ Remove the suffix from string s if present.
    Doing this manually until we start using python 3.9.
    
    >>> remove_suffix("bob.com", ".com")
    'bob'
    >>> remove_suffix("Henrietta", "match")
    'Henrietta'
    """
    i = s.rfind(suffix)
    if i == -1:
        # return s if not found.
        return s
    return s[:i]


def remove_prefix(s: str, prefix: str) -> str:
    """ Remove the prefix from string s if present.
    Doing this manually until we start using python 3.9.
    
    >>> remove_prefix("bob.com", "bo")
    'b.com'
    >>> remove_prefix("Henrietta", "match")
    'Henrietta'
    """
    if not s.startswith(prefix):
        return s
    return s[len(prefix) :]


def get_all_subclasses_of(cls: Type[T]) -> Iterable[Type[T]]:
    scope_dict: Dict = globals()
    for name, var in scope_dict.items():
        if isclass(var) and issubclass(var, cls):
            yield var


def get_all_concrete_subclasses_of(cls: Type[T]) -> Iterable[Type[T]]:
    yield from filterfalse(inspect.isabstract, get_all_subclasses_of(cls))


def get_path_to_source_file(cls: Type) -> Path:
    """ Attempts to give a relative path to the given source path. If not possible, then
    gives back an absolute path to the source file instead.
    """
    cwd = Path.cwd()
    source_file = getsourcefile(cls)
    assert isinstance(source_file, str), f"can't locate source file for {cls}?"
    source_path = Path(source_file).absolute()
    try:
        return source_path.relative_to(cwd)
    except ValueError:
        # If we can't find the relative path, for instance when sequoia is
        # installed in site_packages (not with `pip install -e .``), give back
        # the absolute path instead.
        return source_path


def constant_property(fixed_value: T) -> T:
    def constant_field(v: T, **kwargs) -> T:
        metadata = kwargs.setdefault("metadata", {})
        metadata["constant"] = v
        metadata["decoding_fn"] = lambda _: v
        metadata["to_dict"] = lambda _: v
        return field(default=v, init=False, **kwargs)

    def setter(_, value: Any):
        if isinstance(value, property):
            # This happens in the __init__ that is generated by dataclasses, so we
            # do nothing here.
            pass
        elif value != fixed_value:
            raise RuntimeError(
                RuntimeWarning(f"This attribute is fixed at value {fixed_value}.")
            )

    def getter(_) -> T:
        return fixed_value

    return property(fget=getter, fset=setter)


def deprecated_property(old_name: str, new_name: str):
    """ Marks a property as being deprecated, redirectly any changes to its value to the
    property with name 'new_name'.
    """

    def setter(self, value: Any):
        warnings.warn(
            DeprecationWarning(
                f"'{old_name}' property is deprecated, use '{new_name}' instead."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(value, property):
            # This happens in the __init__ that is generated by dataclasses, so we
            # do nothing here.
            pass
        else:
            setattr(self, new_name, value)
        # raise RuntimeError(f"'{old_name}' property is deprecated, use '{new_name}' instead.")

    def getter(self):
        warnings.warn(
            DeprecationWarning(
                f"'{old_name}' property is deprecated, use '{new_name}' instead."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)

    doc = f"Deprecated property, Please use '{new_name}' instead."
    return property(fget=getter, fset=setter, doc=doc)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
