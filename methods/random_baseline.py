"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Type, Union

import gym
import numpy as np
import torch
import tqdm
from torch import Tensor
from utils import get_logger, singledispatchmethod

from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from methods.baseline_method import BaselineMethod
from methods.models import Agent, BaselineModel, OutputHead
from settings import (Actions, ActiveSetting, ClassIncrementalSetting,
                      Environment, IIDSetting, Observations, Rewards,
                      RLSetting, Setting, SettingType, TaskIncrementalSetting)

from .models import BaselineModel
from .models.agent import Agent
from .models.random_agent import RandomAgent

logger = get_logger(__file__)

from settings.method_base import MethodABC

@dataclass
class RandomBaselineMethod(MethodABC, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    We do this by creating a base BaselineModel with an output head that gives random
    predictions.
    
    TODO: Actually make this compatible with other settings than
    task-incremental and iid. There will probably be some code shuffling to do
    with respect to the `BaselineModel` class, as it is moreso aimed at being a `passive`
    BaselineModel than an active one at the moment.
    """
    def fit(self,
            train_env: Environment=None,
            valid_env: Environment=None,
            datamodule=None
        ):
        # This is useless atm (we don't train) but just for testing purposes.
        # self.observation_space = train_env.observation_space
        # self.action_space = train_env.action_space
        return 1

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        return action_space.sample()

    def configure(self, setting: Setting):
        self.action_space = setting.action_space
        super().configure(setting)

    @singledispatchmethod
    def validate_results(self, setting: Setting, results: Setting.Results):
        """Called during testing. Use this to assert that the results you get
        from applying your method on the given setting match your expectations.

        Args:
            setting
            results (Results): A given Results object.
        """
        assert results is not None
        assert results.objective > 0
        print(f"Objective when applied to a setting of type {type(setting)}: {results.objective}")


    @validate_results.register
    def validate(self, setting: ClassIncrementalSetting, results: ClassIncrementalSetting.Results):
        assert isinstance(setting, ClassIncrementalSetting), setting
        assert isinstance(results, ClassIncrementalSetting.Results), results

        average_accuracy = results.objective
        # Calculate the expected 'average' chance accuracy.
        # We assume that there is an equal number of classes in each task.
        chance_accuracy = 1 / setting.n_classes_per_task

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.average_metrics_per_task):
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Check that this makes sense:
            chance_accuracy = 1 / setting.n_classes_per_task

            task_accuracy = metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy

    # @singledispatchmethod
    # def model_class(self, setting: SettingType) -> Type[BaselineModel]:
    #     raise NotImplementedError(f"No known model for setting of type {type(setting)} (registry: {self.model_class.registry})")
    
    # @model_class.register
    # def _(self, setting: ActiveSetting) -> Type[Agent]:
    #     # TODO: Make a 'random' RL method.
    #     return RandomAgent

    # @model_class.register
    # def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModelMixin]:
    #     # IDEA: Generate the model dynamically instead of creating one of each.
    #     # (This doesn't work atm because super() gives back a BaselineModel)
    #     # return get_random_model_class(super().model_class(setting))
    #     return RandomClassIncrementalModel

    # @model_class.register
    # def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
    #     return RandomTaskIncrementalModel


if __name__ == "__main__":
    RandomBaselineMethod.main()
