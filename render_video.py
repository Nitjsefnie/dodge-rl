"""Render DodgeHumanoid-v0 episodes to an mp4, headless.

GL backend: MUJOCO_GL must be set BEFORE the first mujoco import in the
process (mujoco binds its GL backend at import). This script honours an
already-set MUJOCO_GL; otherwise it defaults to "osmesa". If osmesa fails to
initialize, the script re-execs itself (os.execve) exactly once with
MUJOCO_GL=egl — re-importing mujoco in the same process is not possible, so a
re-exec is the only safe retry. A GL init failure manifests at import or at
the first env.render(); the retry is therefore scoped to that phase: once the
first render has succeeded, later exceptions are real bugs and are never
retried or misattributed to the GL backend. The backend that worked is
printed to stdout.

Rendering route: gymnasium.make("DodgeHumanoid-v0", render_mode="rgb_array",
width=640, height=480, camera_id=-1, default_camera_config=...) and
env.render() per captured frame — MujocoEnv's built-in offscreen pipeline.
camera_id=-1 selects the free camera (the XML's "track" camera would
otherwise win); default_camera_config then sets a fixed free camera framing
the 3x3 arena (distance 7.0, elevation -15, azimuth 90, lookat (0, 0, 1.0)).
Verified against the installed gymnasium 1.3.0 source
(gymnasium/envs/mujoco/mujoco_env.py, mujoco_rendering.py).

--fps 33 means: capture every 2nd control step (67 Hz control / 2 ~ 33.4
fps) and write the mp4 with fps=33 via imageio-ffmpeg. The control rate is
read from env.metadata["render_fps"] (67 as fallback). Frames are streamed
to the writer as they are captured; they are never accumulated in memory.

Usage:
    python render_video.py --checkpoint runs/.../final.zip [--episodes 2]
        [--out demo.mp4] [--fps 33] [--seed 0]
    python render_video.py --random [--episodes 2] [--out demo.mp4]
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

DEFAULT_CONTROL_FPS = 67  # fallback if env.metadata lacks render_fps
CAMERA_CONFIG = {
    "distance": 7.0,
    "elevation": -15.0,
    "azimuth": 90.0,
    "lookat": (0.0, 0.0, 1.0),
}
_RETRY_FLAG = "RENDER_VIDEO_GL_RETRIED"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to an SB3 PPO checkpoint (.zip).")
    policy.add_argument("--random", action="store_true",
                        help="Render a uniform random policy.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("demo.mp4"))
    parser.add_argument("--fps", type=int, default=33)
    parser.add_argument("--seed", type=int, default=0)
    # Hidden: stop after this many captured frames (None = unlimited).
    parser.add_argument("--max-frames", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    return args


def _gl_retry_available():
    return (os.environ.get("MUJOCO_GL") != "egl"
            and not os.environ.get(_RETRY_FLAG))


def _reexec_with_egl():
    """Re-exec this script once with MUJOCO_GL=egl. Does not return."""
    print("GL backend failed before first render; retrying with MUJOCO_GL=egl",
          file=sys.stderr)
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env[_RETRY_FLAG] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def run(args, gl_state):
    import imageio.v2 as imageio
    import numpy as np

    import gymnasium

    import dodge_rl  # noqa: F401  (registers DodgeHumanoid-v0)

    model = None
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            print(f"error: checkpoint not found: {args.checkpoint}",
                  file=sys.stderr)
            return 3
        from stable_baselines3 import PPO

        model = PPO.load(str(args.checkpoint), device="cpu")

    camera_config = dict(CAMERA_CONFIG)
    camera_config["lookat"] = np.asarray(camera_config["lookat"])

    env = gymnasium.make(
        "DodgeHumanoid-v0",
        render_mode="rgb_array",
        width=640,
        height=480,
        camera_id=-1,
        default_camera_config=camera_config,
    )
    env.action_space.seed(args.seed)

    control_fps = env.metadata.get("render_fps", DEFAULT_CONTROL_FPS)
    capture_every = max(1, round(control_fps / args.fps))
    frames_written = 0
    episode_summaries = []

    def capped():
        return args.max_frames is not None and frames_written >= args.max_frames

    def capture():
        nonlocal frames_written
        imageio_writer.append_data(env.render())
        # First successful render proves the GL backend is up; failures past
        # this point are not GL-init failures and must not trigger a re-exec.
        gl_state["rendered"] = True
        frames_written += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio_writer = imageio.get_writer(args.out, fps=args.fps)
    try:
        for ep in range(args.episodes):
            if capped():
                break
            obs, _ = env.reset(seed=args.seed if ep == 0 else None)
            capture()
            total_reward, steps = 0.0, 0
            info = {}
            while not capped():
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if steps % capture_every == 0:
                    capture()
                if terminated or truncated:
                    break
            if steps > 0:  # cap landing on an episode boundary: nothing ran
                episode_summaries.append(
                    f"episode {ep}: reward={total_reward:.2f} "
                    f"hits={info.get('hits', 0)} "
                    f"wall_death={info.get('wall_death', False)}"
                )
    finally:
        imageio_writer.close()
        env.close()

    print(f"backend: {os.environ.get('MUJOCO_GL')}")
    print(f"frames written: {frames_written}")
    print(f"output: {args.out}")
    for line in episode_summaries:
        print(line)
    return 0


def main(argv=None):
    args = parse_args(argv)

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"

    gl_state = {"rendered": False}
    try:
        return run(args, gl_state)
    except Exception:
        # Retry only GL-init failures (import-time or first render); once a
        # frame has been rendered the backend works and the exception is a
        # real bug — report it as-is, never re-exec.
        if not gl_state["rendered"] and _gl_retry_available():
            _reexec_with_egl()
        traceback.print_exc()
        print("error: render_video.py failed (see traceback above)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
