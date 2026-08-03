"""Smoke test for the PPO training script."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


RUN_DIR = Path("runs/smoke")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_smoke():
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    cmd = [
        sys.executable,
        "train.py",
        "--total-steps",
        "4096",
        "--n-envs",
        "2",
        "--seed",
        "3",
        "--run-name",
        "smoke",
    ]
    result = subprocess.run(
        ["nice", "-n", "10"] + cmd,
        cwd=REPO_ROOT,
        timeout=600,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"train.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    monitor_csv = RUN_DIR / "monitor.csv"
    assert monitor_csv.exists(), "monitor.csv missing"
    lines = monitor_csv.read_text().strip().splitlines()
    assert len(lines) >= 2, "monitor.csv should have header plus >=1 data row"

    zips = list(RUN_DIR.glob("final.zip")) + list(RUN_DIR.glob("ckpt_*.zip"))
    assert zips, "expected final.zip or a ckpt_*.zip checkpoint"

    shutil.rmtree(RUN_DIR)
