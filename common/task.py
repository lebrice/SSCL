from dataclasses import dataclass, field
from typing import List

from simple_parsing import list_field

from utils.json_utils import Serializable


@dataclass
class Task(Serializable):
    """ Dataclass that represents a task.

    TODO: Maybe the this could also specify from which dataset(s) it is sampled.
    """
    # The index of this task (the order in which it was encountered)
    index: int = field(default=-1, repr=False)
    # All the unique classes present within this task. (order matters)
    classes: List[int] = list_field()
