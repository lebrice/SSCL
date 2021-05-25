from sequoia.settings.sl.continual import ContinualSLSetting
from dataclasses import dataclass
from sequoia.utils.utils import constant, flag
from sequoia.settings.assumptions.context_discreteness import DiscreteContextAssumption


@dataclass
class DiscreteTaskAgnosticSLSetting(DiscreteContextAssumption, ContinualSLSetting):
    """ Continual Supervised Learning Setting where there are clear task boundaries, but
    where the task information isn't available.
    """
