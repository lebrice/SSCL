"""Runs an experiment, which consist in applying a Method to a Setting.


"""
import inspect
import json
import shlex
import sys
import traceback
from argparse import Namespace
from collections import OrderedDict
from dataclasses import InitVar, dataclass
from inspect import isabstract, isclass
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from simple_parsing import (ArgumentParser, choice, field, mutable_field,
                            subparsers)

from methods import Method, MethodType, all_methods
from settings import (ClassIncrementalResults, Results, Setting, SettingType,
                      all_settings)
from utils import Parseable, Serializable, get_logger

logger = get_logger(__file__)

logger.debug(f"Registered Settings: \n" + "\n".join(
    f"- {setting.get_name()}: {setting}" for setting in all_settings
))

logger.debug(f"Registered Methods: \n" + "\n".join(
    f"- {method.get_name()}: {method.target_setting} {method}" for method in all_methods
))

@dataclass
class Experiment(Parseable, Serializable):
    """ Applies a Method to an experimental Setting to obtain Results.

    When the `setting` is not set, calling `launch` on the
    `Experiment` will evaluate the chosen method on all "applicable" settings. 
    (i.e. lower in the Settings inheritance tree).

    When the `method` is not set, this will apply all applicable methods on the
    chosen setting.
    """
    # Which experimental setting to use. When left unset, will evaluate the
    # provided method on all applicable settings.
    setting: Optional[Union[str, Setting, Type[Setting]]] = choice(
        {setting.get_name(): setting for setting in all_settings},
        default=None,
    )

    # Which experimental method to use. When left unset, will evaluate all
    # compatible methods on the provided setting.
    # NOTE: Some methods can share the same name, for instance, 'baseline' may
    # refer to the ClassIncrementalMethod or TaskIncrementalMethod.
    # Therefore, the given `method` is a string (for example when creating this
    # class from the command-line) and there are multiple methods with the given
    # name, then the most specific method applicable for the given setting will
    # be used.
    method: Optional[Union[str, Method, Type[Method]]] = choice(
        set(method.get_name() for method in all_methods),
        default=None,
    )

    def __post_init__(self):
        if not (self.setting or self.method):
            raise RuntimeError(
                "At least one of `setting` or `method` must be set!"
            )
        if isinstance(self.setting, str):
            # All settings must have a unique name.
            settings_with_that_name: List[Type[Setting]] = [
                setting for setting in all_settings
                if setting.get_name() == self.setting
            ]
            if not settings_with_that_name:
                raise RuntimeError(
                    f"No settings found with name '{self.setting}'!"
                    f"Available settings : \n" + "\n".join(
                        f"- {setting.get_name()}: {setting}"
                        for setting in all_settings
                    )
                )
            elif len(settings_with_that_name) == 1:
                self.setting = settings_with_that_name[0]
            else:
                raise RuntimeError(
                    f"Error: There are multiple settings with the same name, "
                    f"which isn't allowed! (name: {self.setting}, culprits: "
                    f"{settings_with_that_name})"
                )
        

    def launch(self, argv: Union[str, List[str]] = None) -> Results:
        if isclass(self.setting) and issubclass(self.setting, Setting):
            self.setting = self.setting.from_args(argv)

        if isclass(self.method) and issubclass(self.method, Method):
            self.method = self.method.from_args(argv)

        potential_methods: List[Type[Method]] = []
        method_name: Optional[str] = self.method if isinstance(self.method, str) else self.method.get_name() 
        
        if isinstance(self.method, str):
            # Collisions in method names should be allowed. If it happens, we shoud
            # use the right method for the given setting, if any.
            # There's also the special case where only a method string was given!
            # What should we do in that case?
            potential_methods: List[Type[Method]] = [
                method for method in all_methods
                if method.get_name() == self.method
            ]
            if self.setting:
                potential_methods = [m for m in potential_methods if m.is_applicable(self.setting)]

            if not potential_methods:
                raise RuntimeError(
                    f"Couldn't find any methods with name {self.method} "
                    f"applicable on the chosen setting ({self.setting})!"
                )
            if len(potential_methods) == 1:
                self.method = potential_methods[0]
                return self.setting.apply(self.method)
            else:
                # TODO: figure out which of the methods to use depending on the setting?
                raise NotImplementedError("TODO")
                logger.warning(RuntimeWarning(
                    f"As there are multiple methods with the name {self.method}, "
                    f"this will try to use the most 'specialised' method with that "
                    f"name for the given setting. (potential methods: "
                    f"{potential_methods}"
                ))
                    



        assert self.setting is None or isinstance(self.setting, Setting)

        if self.setting:
            if isinstance(self.method, Method):
                return self.setting.apply(self.method)
            else:
                # When the method isn't set, evaluate on all applicable methods.
                all_results: Dict[Type[Method], Results] = OrderedDict()

                for setting_type in self.setting.get_all_applicable_methods():
                    setting = setting_type.from_args(argv)
                    results = setting.apply(self)
                    all_results[setting_type] = results

                logger.info(f"All results for setting of type {type(self)}:")
                logger.info({
                    setting.get_name(): (results if results else "crashed")
                    for setting, results in all_results.items()
                })
                return all_results

        assert self.method
        if isinstance(self.method, Method):
            return self.method.apply_to(self.setting)
        elif isinstance(self.method, str):
            raise NotImplementedError("TODO")
        else:
            # When the method isn't set, evaluate on all applicable methods.
            all_results: Dict[Type[Method], Results] = OrderedDict()

            for setting_type in self.method.get_all_applicable_settings():
                setting = setting_type.from_args(argv)
                results = self.method.apply_to(setting)
                all_results[setting_type] = results
            logger.info(f"All results for method of type {type(self.method)}:")
            logger.info({
                setting.get_name(): (results.get_metric() if results else "crashed")
                for setting, results in all_results.items()
            })


    @classmethod
    def main(cls, argv: Union[str, List[str]] = None) -> Union[Results, Dict[Type[Setting], Results], Dict[Type[Method], Results]]:
        """Launches an experiment using the given command-line arguments.

        First, we get the choice of method and setting using a first parser.
        Then, we parse the Setting and Method objects using the remaining args
        with two other parsers.

        Parameters
        ----------
        - argv : Union[str, List[str]], optional, by default None

            command-line arguments to use. When None (default), uses sys.argv.

        Returns
        -------
        Results
            Results of the experiment.
        """
        experiment, unused_args = cls.from_known_args(argv)
        experiment: Experiment
        return experiment.launch(unused_args)


if __name__ == "__main__":
    results = Experiment.main()
    if results:
        # Experiment didn't crash, show results:
        print(f"Objective: {results.objective}")
        # print(f"Results: {results}")
        if isinstance(results, ClassIncrementalResults):
            print(f"task metrics:")
            for m in results.task_metrics:
                print(m)
