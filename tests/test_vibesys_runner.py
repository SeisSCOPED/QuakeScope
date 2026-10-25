"""The exported benchmark remains runnable and safe on CPUs without AMX."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workspace_layout", [False, True])
def test_runner_help_from_both_layouts(tmp_path, workspace_layout):
    runner = ROOT / "vibesys" / "pick_workload.py"
    if workspace_layout:
        shutil.copyfile(runner, tmp_path / "pick_workload.py")
        (tmp_path / "quakescope").symlink_to(ROOT, target_is_directory=True)
        runner = tmp_path / "pick_workload.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--help"], cwd=tmp_path,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--workload" in result.stdout


@pytest.mark.parametrize("has_amx, expected", [(False, "fp32"), (True, "bf16")])
def test_bf16_requires_amx(monkeypatch, has_amx, expected):
    from sb_catalog.src import picker

    monkeypatch.setattr(picker, "amx_bf16_available", lambda: has_amx)
    assert picker.resolve_precision("bf16") == expected
    assert picker.resolve_precision("fp32") == "fp32"


def test_amx_can_be_disabled(monkeypatch):
    from sb_catalog.src import picker

    monkeypatch.setenv("QS_AMX", "off")
    assert not picker.amx_bf16_available()
    assert picker.resolve_precision("bf16") == "fp32"
