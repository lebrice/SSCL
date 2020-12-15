from typing import NamedTuple

from ._namedtuple import is_namedtuple, is_namedtuple_type

class DummyTuple(NamedTuple):
    a: int
    b: str

def test_is_namedtuple():
    bob = DummyTuple(1, "bob")
    assert is_namedtuple(bob)

def test_is_namedtuple_type():
    assert is_namedtuple_type(DummyTuple)
    assert is_namedtuple_type(NamedTuple)
    assert not is_namedtuple_type(tuple)
    assert not is_namedtuple_type(list)
    assert not is_namedtuple_type(dict)


def test_instance_check():
    bob = DummyTuple(1, "bob")
    assert isinstance(bob, DummyTuple)
    assert isinstance(bob, NamedTuple)
    assert isinstance(bob, tuple)


def test_instance_check():
    from typing import Tuple
    assert issubclass(DummyTuple, NamedTuple)
    assert issubclass(DummyTuple, tuple)
    assert issubclass(DummyTuple, DummyTuple)
    assert not issubclass(list, DummyTuple)
    assert not issubclass(tuple, DummyTuple)
    assert not issubclass(NamedTuple, DummyTuple)

