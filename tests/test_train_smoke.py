"""Smoke tests for the PPO training script."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "smoke"


def _run_train(args, timeout=600):
    """Run train.py under nice -n 10 and return the CompletedProcess."""
    cmd = [sys.executable, "train.py"] + args
    return subprocess.run(
        ["nice", "-n", "10"] + cmd,
        cwd=REPO_ROOT,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_train_smoke():
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    try:
        result = _run_train(
            [
                "--total-steps",
                "4096",
                "--n-envs",
                "2",
                "--seed",
                "3",
                "--run-name",
                "smoke",
            ]
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
    finally:
        if RUN_DIR.exists():
            shutil.rmtree(RUN_DIR)


@pytest.mark.slow
def test_refuses_to_clobber_existing_run():
    """A run dir containing monitor.csv without --resume is rejected with exit 2."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        run_dir = out_dir / "clobber"
        run_dir.mkdir(parents=True)
        (run_dir / "monitor.csv").write_text("# dummy\n")

        result = _run_train(
            [
                "--out-dir",
                str(out_dir),
                "--run-name",
                "clobber",
                "--total-steps",
                "4096",
                "--n-envs",
                "2",
                "--seed",
                "1",
            ]
        )
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.slow
def test_sigterm_saves_interrupt_and_resume_preserves_monitor():
    """SIGTERM saves interrupt.zip; resuming renames the old monitor.csv."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        run_dir = out_dir / "sigterm"

        if run_dir.exists():
            shutil.rmtree(run_dir)

        proc = None
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "train.py",
                    "--out-dir",
                    str(out_dir),
                    "--run-name",
                    "sigterm",
                    "--total-steps",
                    "40960",
                    "--n-envs",
                    "2",
                    "--seed",
                    "7",
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=lambda: os.nice(10),
            )

            monitor_csv = run_dir / "monitor.csv"
            # Poll until at least one episode has been written.
            # VecMonitor writes two header lines, so we need >= 3 lines.
            deadline = time.time() + 180
            while time.time() < deadline:
                if monitor_csv.exists():
                    lines = monitor_csv.read_text().strip().splitlines()
                    if len(lines) >= 3:
                        break
                time.sleep(0.25)
            else:
                proc.terminate()
                proc.wait(timeout=30)
                pytest.fail(
                    "monitor.csv did not gain a data row before timeout; "
                    f"stdout:\n{proc.stdout.read() if proc.stdout else ''}\n"
                    f"stderr:\n{proc.stderr.read() if proc.stderr else ''}"
                )

            proc.send_signal(signal.SIGTERM)

            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)

            assert proc.returncode != 0, (
                f"expected nonzero exit after SIGTERM, got {proc.returncode}\n"
                f"stdout:\n{proc.stdout.read() if proc.stdout else ''}\n"
                f"stderr:\n{proc.stderr.read() if proc.stderr else ''}"
            )
            assert (run_dir / "interrupt.zip").exists(), "interrupt.zip missing"

            # Resume from the interrupted checkpoint.
            result = _run_train(
                [
                    "--out-dir",
                    str(out_dir),
                    "--run-name",
                    "sigterm",
                    "--resume",
                    str(run_dir / "interrupt.zip"),
                    "--total-steps",
                    "4096",
                    "--n-envs",
                    "2",
                    "--seed",
                    "7",
                ]
            )
            assert result.returncode == 0, (
                f"resume run exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

            backups = list(run_dir.glob("monitor.csv.*.bak"))
            assert backups, "pre-resume monitor.csv was not backed up"
            backup_lines = backups[0].read_text().strip().splitlines()
            assert len(backup_lines) >= 2, "backup should contain pre-resume data rows"
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
            if run_dir.exists():
                shutil.rmtree(run_dir)
