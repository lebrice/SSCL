import pytest
from sequoia.common.config import Config
from sequoia.conftest import monsterkong_required
from sequoia.settings.active import (
    IncrementalRLSetting, RLSetting
)

from .ppo import PPOMethod, PPOModel


def test_cartpole_state():
    method = PPOMethod(hparams=PPOModel.HParams(n_steps=64))
    setting = RLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        steps_per_task=5_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())
    assert 150 < results.average_final_performance.mean_reward_per_episode


def test_incremental_cartpole_state():
    method = PPOMethod(hparams=PPOModel.HParams(n_steps=64))
    setting = IncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())


@pytest.mark.timeout(120)
@monsterkong_required
def test_monsterkong():
    method = PPOMethod()
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())
