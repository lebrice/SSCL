""" WIP: Tests for the AGEM Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .agem import AGEMMethod
from .base_test import _TestAvalancheMethod


class TestAGEMMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = AGEMMethod
