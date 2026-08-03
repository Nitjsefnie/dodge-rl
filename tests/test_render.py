"""Render smoke test: render_video.py --random with a 30-frame cap."""

import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = REPO_ROOT / "render_video.py"


@pytest.mark.slow
def test_render_random_30_frames(tmp_path):
    out = tmp_path / "render_test.mp4"
    result = subprocess.run(
        [
            "nice", "-n", "10",
            sys.executable, str(RENDER_SCRIPT),
            "--random",
            "--episodes", "2",
            "--max-frames", "30",
            "--seed", "0",
            "--out", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"render_video.py failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "backend:" in result.stdout, (
        f"missing backend line in stdout:\n{result.stdout}"
    )
    assert out.is_file(), f"output not written: {out}"
    assert out.stat().st_size > 10 * 1024, (
        f"output too small: {out.stat().st_size} bytes"
    )
    with imageio.get_reader(out) as reader:
        n_frames = reader.count_frames()
        frame = reader.get_data(0)
    assert n_frames >= 30, f"expected >= 30 frames, got {n_frames}"
    assert frame.max() > frame.min(), "frame 0 is degenerate (uniform color)"
