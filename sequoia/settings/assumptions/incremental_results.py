""" Results of an Incremental setting. """
import json
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import wandb
from gym.utils import colorize
from sequoia.common.metrics import Metrics
from sequoia.settings.base.results import Results
from simple_parsing.helpers import list_field
from simple_parsing.helpers.serialization import encode

from .iid_results import MetricType, TaskResults
from .discrete_results import TaskSequenceResults


@dataclass
class IncrementalResults(Results, Generic[MetricType]):
    """ Results for a whole train loop (transfer matrix).

    This class is basically just a 2d list of TaskResults objects, with some convenience
    methods and properties.
    We get one TaskSequenceResults (a 1d list of TaskResults objects) as a result of
    every test loop, which, in the Incremental Settings, happens after training on each
    task, hence why we get a nb_tasks x nb_tasks matrix of results.
    """

    task_sequence_results: List[TaskSequenceResults[MetricType]] = list_field()

    min_runtime_hours: ClassVar[float] = 0.0
    max_runtime_hours: ClassVar[float] = 12.0

    def __post_init__(self):
        self._runtime: Optional[float] = None
        self._online_training_performance: Optional[List[Dict[int, Metrics]]] = None
        # Factor used to scale the 'objective' to a 'score' between 0 and 1.
        self._objective_scaling_factor: float = 1.0

    @property
    def runtime_minutes(self) -> Optional[float]:
        return self._runtime / 60 if self._runtime is not None else None

    @property
    def runtime_hours(self) -> Optional[float]:
        return self._runtime / 3600 if self._runtime is not None else None

    @property
    def transfer_matrix(self) -> List[List[TaskResults]]:
        return [
            task_sequence_result.task_results for task_sequence_result in 
            self.task_sequence_results
        ]

    @property
    def metrics_matrix(self) -> List[List[MetricType]]:
        """Returns the 'transfer matrix' but with the average metrics for each task
        in each cell.

        NOTE: This is different from `transfer_matrix` since it returns the matrix of
        `TaskResults` objects (which are themselves lists of Metrics) 

        Returns
        -------
        List[List[MetricType]]
            2d grid of average metrics for each task.
        """
        return [
            [task_results.average_metrics for task_results in task_sequence_result]
            for task_sequence_result in self
        ]

    @property
    def objective_matrix(self) -> List[List[float]]:
        """Return transfer matrix containing the value of the 'objective' for each task.

        The value at the index (i, j) gives the test performance on task j after having
        learned tasks 0-i.

        Returns
        -------
        List[List[float]]
            The 2d matrix of objectives (floats).
        """
        return [
            [task_result.objective for task_result in task_sequence_result]
            for task_sequence_result in self.transfer_matrix
        ]

    @property
    def cl_score(self) -> float:
        """ CL Score, as a weigted sum of three objectives:
        - The average final performance over all tasks
        - The average 'online' performance over all tasks
        - Runtime

        TODO: @optimass Determine the weights for each factor.

        Returns
        -------
        float
            [description]
        """
        # TODO: Determine the function to use to get a runtime score between 0 and 1.
        score = (
            +0.30 * self._online_performance_score()
            + 0.40 * self._final_performance_score()
            + 0.30 * self._runtime_score()
        )
        return score

    def _runtime_score(self) -> float:
        # TODO: function that takes the total runtime in seconds and returns a
        # normalized float score between 0 and 1.
        runtime_seconds = self._runtime
        if self._runtime is None:
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Runtime is None! Returning runtime score of 0.\n (Make sure the "
                        "Setting had its `monitor_training_performance` attr set to True!",
                        color="red",
                    )
                )
            )
            return 0
        runtime_hours = runtime_seconds / 3600

        # Get the maximum runtime for this type of Results (and Setting)
        min_runtime_hours = type(self).min_runtime_hours
        max_runtime_hours = type(self).max_runtime_hours

        assert 0 <= min_runtime_hours < max_runtime_hours
        assert 0 < runtime_hours
        if runtime_hours <= min_runtime_hours:
            return 1.0
        if max_runtime_hours <= runtime_hours:
            return 0.0
        return 1 - (
            (runtime_hours - min_runtime_hours)
            / (max_runtime_hours - min_runtime_hours)
        )

    def _online_performance_score(self) -> float:
        """Function that takes the 'objective' of the Metrics from the average online
        performance, and returns a normalized float score between 0 and 1.
        """
        objectives: List[float] = [
            task_online_metric.objective
            for task_online_metric in self.online_performance_metrics
        ]
        return self._objective_scaling_factor * np.mean(objectives)
        # return self._objective_scaling_factor * self.average_online_performance.objective

    def _final_performance_score(self) -> float:
        """ Function that takes the 'objective' of the Metrics from the average
        final performance, and returns a normalized float score between 0 and 1.
        """
        objectives: List[float] = [
            task_metric.objective for task_metric in self.final_performance_metrics
        ]
        return self._objective_scaling_factor * np.mean(objectives)
        # return self._objective_scaling_factor * self.average_final_performance.objective

    @property
    def objective(self) -> float:
        # return self.cl_score
        return self.average_final_performance.objective

    @property
    def num_tasks(self) -> int:
        return len(self.task_sequence_results)

    @property
    def online_performance(self) -> List[Dict[int, MetricType]]:
        """ Returns the online training performance for each task. i.e. the diagonal of
        the transfer matrix.
        
        In SL, this is only recorded over the first epoch.

        Returns
        -------
        List[Dict[int, MetricType]]
            A List containing, for each task, a dictionary mapping from step number to
            the Metrics object produced at that step.
        """
        if not self._online_training_performance:
            return [{} for _ in range(self.num_tasks)]
        return self._online_training_performance

        # return [self[i][i] for i in range(self.num_tasks)]

    @property
    def online_performance_metrics(self) -> List[MetricType]:
        return [
            sum(online_performance_dict.values(), Metrics())
            for online_performance_dict in self.online_performance
        ]

    @property
    def final_performance(self) -> List[TaskResults[MetricType]]:
        return self.transfer_matrix[-1]

    @property
    def final_performance_metrics(self) -> List[MetricType]:
        return [task_result.average_metrics for task_result in self.final_performance]

    @property
    def average_online_performance(self) -> MetricType:
        return sum(self.online_performance_metrics, Metrics())

    @property
    def average_final_performance(self) -> MetricType:
        return sum(self.final_performance_metrics, Metrics())

    def to_log_dict(self, verbose: bool = False) -> Dict:
        log_dict = {}
        # TODO: This assumes that the metrics were stored in the right index for their
        # corresponding task.
        for task_id, task_sequence_result in enumerate(self.task_sequence_results):
            log_dict[f"Task {task_id}"] = task_sequence_result.to_log_dict(
                verbose=verbose
            )

        if self._online_training_performance:
            log_dict["Online Performance"] = {
                f"Task {task_id}": task_online_metrics.to_log_dict(verbose=verbose)
                for task_id, task_online_metrics in enumerate(
                    self.online_performance_metrics
                )
            }

        log_dict.update(
            {
                "Final/Average Online Performance": self._online_performance_score(),
                "Final/Average Final Performance": self._final_performance_score(),
                "Final/Runtime (seconds)": self._runtime,
                "Final/CL Score": self.cl_score,
            }
        )
        return log_dict

    def summary(self, verbose: bool = False):
        s = StringIO()
        log_dict = self.to_log_dict(verbose=verbose)
        log_dict_json = json.dumps(log_dict, indent="\t", default=encode) 
        print(log_dict_json, file=s)
        s.seek(0)
        return s.read()

    def make_plots(self) -> Dict[str, Union[plt.Figure, Dict]]:
        plots = {
            f"Task {task_id}": task_sequence_result.make_plots()
            for task_id, task_sequence_result in enumerate(self.task_sequence_results)
        }
        axis_labels = [f"Task {task_id}" for task_id in range(self.num_tasks)]
        if wandb.run:
            plots["Transfer matrix"] = wandb.plots.HeatMap(
                x_labels=axis_labels,
                y_labels=axis_labels,
                matrix_values=self.objective_matrix,
                show_text=True,
            )
            objective_array = np.asfarray(self.objective_matrix)
            perf_per_step = objective_array.mean(-1)
            table = wandb.Table(
                data=[[i + 1, perf] for i, perf in enumerate(perf_per_step)],
                columns=["# of learned tasks", "Average Test performance on all tasks"],
            )
            plots["Test Performance"] = wandb.plot.line(
                table,
                x="# of learned tasks",
                y="Average Test performance on all tasks",
                title="Test Performance vs # of Learned tasks",
            )
        return plots

    def __str__(self) -> str:
        return self.summary()
