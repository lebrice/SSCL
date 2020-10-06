"""Defines the Results of apply a Method to an IID Setting.  
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from io import StringIO
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

from common import ClassificationMetrics, Loss, Metrics, RegressionMetrics
from settings.base.results import Results
from utils.plotting import PlotSectionLabel, autolabel

from .. import TaskIncrementalResults


@dataclass
class IIDResults(TaskIncrementalResults):
    """Results of applying a Method on an IID Setting.    
    TODO: This should be customized, as it doesn't really make sense to use the
    same plots as in ClassIncremental (there is only one task).
    """
    test_metrics: List[Metrics]

    def __post_init__(self):
        if len(self.test_metrics) > 1:
            self.test_metrics = [self.test_metrics]

    def save_to_dir(self, save_dir: Union[str, Path]) -> None:
        # TODO: Add wandb logging here somehow.
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plots: Dict[str, plt.Figure] = self.make_plots()
        
        # Save the actual 'results' object to a file in the save dir.
        results_json_path = save_dir / "results.json"
        self.save(results_json_path)
        print(f"Saved a copy of the results to {results_json_path}")
        
        print(f"\nPlots: {plots}\n")
        for fig_name, figure in plots.items():
            print(f"fig_name: {fig_name}")
            # figure.show()
            # plt.waitforbuttonpress(10)
            path = (save_dir/ fig_name).with_suffix(".jpg")
            path.parent.mkdir(exist_ok=True, parents=True)
            figure.savefig(path)
            print(f"Saved figure at path {path}")

    def make_plots(self) -> Dict[str, plt.Figure]:
        plots_dict = super().make_plots()
        plots_dict.update({
            "class_accuracies": self.class_accuracies_plot()
        })
        return plots_dict

    def class_accuracies_plot(self):
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        y = self.average_metrics.class_accuracy
        x = list(range(len(y)))
        rects = axes.bar(x, y)
        axes.set_title("Class Accuracy")
        axes.set_xlabel("Class")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        # autolabel(axes, rects)
        return figure

    def summary(self) -> str:
        s = StringIO()
        with redirect_stdout(s):
            print(f"Average Accuracy: {self.average_metrics.accuracy:.2%}")
            for i, class_acc in enumerate(self.average_metrics.class_accuracy):
                print(f"Accuracy for class {i}: {class_acc:.3%}")
        s.seek(0)
        return s.read()

    def to_log_dict(self) -> Dict[str, float]:
        results = {}
        results["objective"] = self.objective
        results.update(self.average_metrics.to_log_dict(verbose=True))
        return results
