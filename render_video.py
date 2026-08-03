"""Render DodgeHumanoid-v0 episodes to an mp4, headless.

GL backend: MUJOCO_GL must be set BEFORE the first mujoco import in the
process (mujoco binds its GL backend at import). This script honours an
already-set MUJOCO_GL; otherwise it defaults to "osmesa". If osmesa fails to
initialize, the script re-execs itself (os.execve) exactly once with
MUJOCO_GL=egl — re-importing mujoco in the same process is not possible, so a
re-exec is the only safe retry. The backend that worked is printed to stdout.

Rendering route: gymnasium.make("DodgeHumanoid-v0", render_mode="rgb_array",
width=640, height=480, camera_id=-1, default_camera_config=...) and
env.render() per captured frame — MujocoEnv's built-in offscreen pipeline.
camera_id=-1 selects the free camera (the XML's "track" camera would
otherwise win); default_camera_config then sets a fixed free camera framing
the 3x3 arena (distance 7.0, elevation -15, azimuth 90, lookat (0, 0, 1.0)).
Verified against the installed gymnasium 1.3.0 source
(gymnasium/envs/mujoco/mujoco_env.py, mujoco_rendering.py).

--fps 33 means: capture every 2nd control step (67 Hz control / 2 ~ 33.4
fps) and write the mp4 with fps=33 via imageio-ffmpeg.

Usage:
    python render_video.py --checkpoint runs/.../final.zip [--episodes 2]
        [--out demo.mp4] [--fps 33] [--seed 0]
    python render_video.py --random [--episodes 2] [--out demo.mp4]
"""

import argparse
import os
import sys
from pathlib import Path

CONTROL_FPS = 67  # DodgeHumanoid-v0 metadata render_fps (dt = 0.015 s)
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
    return parser.parse_args(argv)


def _retry_with_egl():
    """Re-exec this script once with MUJOCO_GL=egl."""
    if os.environ.get("MUJOCO_GL") == "egl" or os.environ.get(_RETRY_FLAG):
        return False
    print("GL backend failed; retrying with MUJOCO_GL=egl", file=sys.stderr)
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env[_RETRY_FLAG] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
    return True  # unreachable


def run(args):
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

    capture_every = max(1, round(CONTROL_FPS / args.fps))
    frames = []
    episode_summaries = []

    def capped():
        return args.max_frames is not None and len(frames) >= args.max_frames

    for ep in range(args.episodes):
        if capped():
            break
        obs, _ = env.reset(seed=args.seed if ep == 0 else None)
        frames.append(env.render())
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
                frames.append(env.render())
            if terminated or truncated:
                break
        episode_summaries.append(
            f"episode {ep}: reward={total_reward:.2f} "
            f"hits={info.get('hits', 0)} wall_death={info.get('wall_death', False)}"
        )

    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=args.fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"backend: {os.environ.get('MUJOCO_GL')}")
    print(f"frames written: {len(frames)}")
    print(f"output: {args.out}")
    for line in episode_summaries:
        print(line)
    return 0


def main(argv=None):
    args = parse_args(argv)

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"

    try:
        return run(args)
    except Exception as exc:  # GL backend init can fail at import or render
        if _retry_with_egl():
            raise  # unreachable
        print(f"error: GL backend failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
