from pathlib import Path

import pytest
from common.metrics import ClassificationMetrics
from settings import TaskIncrementalResults, TaskIncrementalSetting
from simple_parsing import ParsingError



# def test_parsing_hparams_multihead():
#     """Test that parsing the multihead field works as expected. """
#     hp = TaskIncrementalModel.HParams.from_args("")
#     assert hp.multihead

#     with pytest.raises(ParsingError):
#         hp = TaskIncrementalModel.HParams.from_args("--multihead")
#         assert hp.multihead

#     hp = TaskIncrementalModel.HParams.from_args("--multihead=False")
#     assert not hp.multihead

#     hp = TaskIncrementalModel.HParams.from_args("--multihead True")
#     assert hp.multihead

#     hp = TaskIncrementalModel.HParams.from_args("--multihead False")
#     assert not hp.multihead


@pytest.mark.skip(reason="This doesn't really belong here anymore")
def test_fast_dev_run_multihead(tmp_path: Path):
    setting = TaskIncrementalSetting(
        dataset="mnist",
        increment=2,
    )
    method: ClassIncrementalMethod = ClassIncrementalMethod.from_args(f"""
        --debug
        --fast_dev_run
        --default_root_dir {tmp_path}
        --log_dir_root {tmp_path}
        --batch_size 100
    """)
    results: TaskIncrementalResults = method.apply_to(setting)
    metrics = results.task_metrics
    assert metrics
    for metric in metrics:
        if isinstance(metric, ClassificationMetrics):
            assert metric.confusion_matrix.shape == (2, 2)
    
    assert 0.45 <= results
