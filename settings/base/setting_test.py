
from dataclasses import dataclass

import pytest

from methods import Method
from utils import constant

from .setting import Setting


@dataclass
class Setting1(Setting):
    foo: int = 1
    bar: int = 2

    def __post_init__(self):
        print(f"Setting1 __init__ ({self})")
        super().__post_init__()


@dataclass
class Setting2(Setting1):
    bar: int = constant(1)

    def __post_init__(self):
        print(f"Setting2 __init__ ({self})")
        super().__post_init__()




def test_settings_override_with_constant_take_init():
    bob1 = Setting1(foo=3, bar=7)
    assert bob1.foo == 3
    assert bob1.bar == 7
    bob2 = Setting2(foo=4, bar=4)
    assert bob2.bar == 1.0
    assert bob2.foo == 4

def test_init_still_works():
    setting = Setting(val_fraction=0.01)
    assert setting.val_fraction == 0.01

@dataclass
class SettingA(Setting): pass

@dataclass
class SettingA1(SettingA): pass

@dataclass
class SettingA2(SettingA): pass

@dataclass
class SettingB(Setting): pass

class MethodA(Method, target_setting=SettingA): pass


class MethodB(Method, target_setting=SettingB): pass


class CoolGeneralMethod(Method, target_setting=Setting): pass
