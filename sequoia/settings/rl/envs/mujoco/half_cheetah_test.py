from typing import ClassVar, Type, List

from sequoia.conftest import mujoco_required
pytestmark = mujoco_required

from .half_cheetah import HalfCheetahV2Env, ContinualHalfCheetahV2Env
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .modified_mass_test import ModifiedMassEnvTests


@mujoco_required
class TestHalfCheetah(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualHalfCheetahV2Env]] = ContinualHalfCheetahV2Env
