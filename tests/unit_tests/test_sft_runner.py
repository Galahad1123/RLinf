# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT runner deadlines at the remote worker boundary."""

import json
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.runners.sft_runner import SFTRunner


class LocalHandle:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value

    def consume_duration(self):
        return 0.01


class LocalActor:
    def __init__(self, deadline_after_update=False):
        self.updates = 0
        self.saves = []
        self.deadline_after_update = deadline_after_update
        self.runner = None

    def get_max_steps_per_epoch(self):
        return LocalHandle([100])

    def run_training(self):
        self.updates += 1
        if self.deadline_after_update:
            self.runner.cfg.runner.deadline_unix = time.time() - 1
        return LocalHandle([{"loss": 1.0}])

    def run_eval(self):
        return LocalHandle([{"val_loss": 1.0}])

    def save_checkpoint(self, path, step):
        self.saves.append(step)
        Path(path, "saved").write_text(str(step))
        return LocalHandle()


def make_runner(tmp_path, deadline=None, stop=False):
    cfg = OmegaConf.create(
        {
            "runner": {
                "max_steps": 3,
                "max_epochs": -1,
                "val_check_interval": -1,
                "save_interval": 3000,
                "deadline_unix": deadline,
                "write_training_status": True,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "test",
                    "logger_backends": [],
                },
            }
        }
    )
    actor = LocalActor(stop)
    runner = SFTRunner(cfg, actor)
    actor.runner = runner
    return runner, actor


def test_deadline_saves_actual_step_and_reports_stop(tmp_path):
    runner, actor = make_runner(tmp_path, time.time() + 1000, True)
    runner.run()
    assert actor.updates == 1
    assert actor.saves == [1]
    report = json.loads((tmp_path / "test/training_status.json").read_text())
    assert report["global_step"] == 1
    assert report["stop_reason"] == "deadline"


def test_disabled_deadline_preserves_updates_and_final_save(tmp_path):
    runner, actor = make_runner(tmp_path)
    runner.run()
    assert actor.updates == 3
    assert actor.saves == [3]
    report = json.loads((tmp_path / "test/training_status.json").read_text())
    assert report["stop_reason"] == "max_steps"


def test_expired_deadline_refuses_new_update(tmp_path):
    runner, actor = make_runner(tmp_path, time.time() - 1)
    with pytest.raises(TimeoutError):
        runner.run()
    assert actor.updates == 0
    assert not actor.saves


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1])
def test_invalid_deadline_rejected(tmp_path, deadline):
    with pytest.raises(ValueError):
        make_runner(tmp_path, deadline)


def test_checkpoint_marker_only_after_successful_save(tmp_path):
    runner, actor = make_runner(tmp_path)
    runner.run()
    checkpoint = tmp_path / "test/checkpoints/global_step_3"
    marker = checkpoint / "checkpoint_complete.json"
    assert json.loads(marker.read_text())["global_step"] == 3

    def interrupted_save(path, step):
        raise OSError("interrupted save")

    actor.save_checkpoint = interrupted_save
    with pytest.raises(OSError):
        runner._save_checkpoint()
    assert not marker.exists()


def test_calibration_deadline_does_not_save_disabled_checkpoints(tmp_path):
    runner, actor = make_runner(tmp_path, time.time() + 1000, True)
    runner.cfg.runner.save_interval = -1
    runner.run()
    assert actor.updates == 1
    assert not actor.saves
