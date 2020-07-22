""" Set of Utilities. """
import collections
import functools
import itertools
import operator
import random
import re
from collections import OrderedDict, defaultdict, deque
from collections.abc import MutableMapping
from dataclasses import Field, fields
from itertools import groupby
from pathlib import Path
from typing import (Any, Callable, Deque, Dict, Iterable, List, MutableMapping,
                    Optional, Set, Tuple, TypeVar, Union)

import numpy as np
from torch import Tensor, cuda, nn

cuda_available = cuda.is_available()
gpus_available = cuda.device_count()

T = TypeVar("T")


def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
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


def to_dict_of_lists(list_of_dicts: Iterable[Dict[str, Any]]) -> Dict[str, List[Tensor]]:
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
        assert d.keys() == result.keys(), f"Dict {d} at index {i} does not contain all the keys!"
    return result


def add_prefix(some_dict: Dict[str, T], prefix: str="", sep=" ") -> Dict[str, T]:
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
    import torch
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj: Any, attr: str, *args):
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and len(list(path.iterdir())) > 0

D = TypeVar("D", bound=Dict)

def flatten_dict(d: D, separator: str="/") -> D:
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


def unique_consecutive(iterable: Iterable[T], key: Callable[[T], Any]=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen.
    
    >>> list(unique_consecutive('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_consecutive('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    
    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


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
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def take(iterable: Iterable[T], n: int) -> Iterable[T]:
    """ Takes only the first `n` elements from `iterable`. """
    return itertools.islice(iterable, n)


def camel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2


if __name__ == "__main__":
    import doctest
    doctest.testmod()
