from typing import Type, ClassVar

import pytest

from sequoia.methods.base_method_test import TestBaseMethod as BaseMethodTests
from sequoia.methods.packnet_method import PackNetMethod
from sequoia.methods.base_method import BaseMethod
from sequoia.common.config import Config
from sequoia.methods.trainer import TrainerConfig


class TestPackNetMethod(BaseMethodTests):
    Method: ClassVar[Type[PackNetMethod]] = PackNetMethod

    def validate_results(self, setting, method, results):
        """ Called at the end of each test run to check that the results make sense for
        the given setting and method.
        """ 
        super().validate_results(setting, method, results) 
        # TODO: Add checks to make sure that the packnet callback's state makes sense
        # for the given setting.
