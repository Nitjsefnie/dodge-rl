#!/usr/bin/env python3
"""PPO training script for DodgeHumanoid-v0."""

import argparse
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

# Set before torch is imported: the OpenMP runtime reads these once at library
# init, so assigning them later has no effect. Measured on a clean 4-vCPU
# runner over 3 interleaved rounds (bench/sweep.py, 2026-08-03): 939.5 vs
# 844.6 median steps/s against the previous spin-wait default, a 1.11x gain
# with 1.2% spread and the same ordering in every round. setdefault, so
# bench/sweep.py can still pin these explicitly to A/B them.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


# Intra-op thread cap for the learner's torch ops. Overridable from the
# environment so bench/sweep.py can A/B thread counts across otherwise
# identical processes; the default is what training has always used.
TORCH_NUM_THREADS = int(os.environ.get("DODGE_TORCH_THREADS", "2"))

# The PPO configuration, as a dict so bench/bench_learn.py can build the exact
# same model instead of restating the hyperparameters (a hand-copied config
# drifts from the one that actually runs). ``env`` and ``seed`` are supplied by
# the caller.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[256, 256]),
    n_steps=2048,
    batch_size=4096,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    device="cpu",
    verbose=0,
)


def apply_rollout_optimizations(model) -> None:
    """Install throughput optimizations on an already-built PPO model.

    Kept as a single entry point so the benchmark (bench/bench_learn.py) and
    real training exercise the identical code path. Must never change *what*
    is computed — only how fast it is computed.
    """
    return None


def make_env_factory(seed: int, rank: int):
    """Return an env factory for SubprocVecEnv worker ``rank``."""

    def _init():
        # Editable install guarantees this import works from any CWD.
        import dodge_rl  # noqa: F401  registers DodgeHumanoid-v0

        env = gym.make("DodgeHumanoid-v0")
        env.reset(seed=seed + rank)
        return env

    return _init


class ProgressCallback(BaseCallback):
    """Print concise progress every 50_000 total steps (nearest rollout boundary)."""

    PRINT_INTERVAL = 50_000
    LOOKBACK = 100

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._next_print: int | None = None
        self._episodes: deque[dict] = deque(maxlen=self.LOOKBACK)

    def _on_training_start(self) -> None:
        # Avoid a torrent of progress lines when resuming from N steps:
        # the next STRICTLY greater boundary (resuming at an exact multiple
        # must not reprint it, and 0 rounds up to the first real boundary).
        ts = self.model.num_timesteps
        self._next_print = ((ts // self.PRINT_INTERVAL) + 1) * self.PRINT_INTERVAL

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._episodes.append(ep)

        if self.num_timesteps >= self._next_print:
            self._print_progress()
            self._next_print += self.PRINT_INTERVAL
        return True

    def _print_progress(self) -> None:
        if not self._episodes:
            print(f"Step {self.num_timesteps}: no episodes yet")
            return

        episodes = list(self._episodes)
        mean_r = sum(e["r"] for e in episodes) / len(episodes)
        mean_l = sum(e["l"] for e in episodes) / len(episodes)
        mean_hits = sum(e.get("hits", 0) for e in episodes) / len(episodes)
        wall_deaths = sum(bool(e.get("wall_death", False)) for e in episodes)
        wall_frac = wall_deaths / len(episodes)
        print(
            f"Step {self.num_timesteps}: reward={mean_r:.3f}, "
            f"len={mean_l:.1f}, hits={mean_hits:.2f}, wall_death={wall_frac:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on DodgeHumanoid-v0")
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(TORCH_NUM_THREADS)

    total_steps = args.total_steps
    n_envs = args.n_envs
    seed = args.seed
    run_name = args.run_name or f"ppo-{seed}-{total_steps}"
    run_dir = Path(args.out_dir) / run_name
    monitor_csv = run_dir / "monitor.csv"

    if monitor_csv.exists() and args.resume is None:
        print(f"Refusing to clobber existing run: {run_dir}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)

    # VecMonitor opens monitor.csv in write mode; preserve prior rows on resume.
    if args.resume and monitor_csv.exists():
        backup = run_dir / f"monitor.csv.{time.time_ns()}.bak"
        monitor_csv.rename(backup)

    env_fns = [make_env_factory(seed, i) for i in range(n_envs)]
    vec_env = VecMonitor(
        SubprocVecEnv(env_fns),
        filename=str(monitor_csv),
        info_keywords=("hits", "wall_death", "fall_death", "spawns", "min_approach"),
    )

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, device="cpu")
        set_random_seed(args.seed)
    else:
        model = PPO(env=vec_env, seed=seed, **PPO_KWARGS)

    apply_rollout_optimizations(model)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // n_envs, 1),
        save_path=str(run_dir),
        name_prefix="ckpt",
    )
    progress_callback = ProgressCallback()

    # Convert SIGTERM into a SystemExit so the finally block saves interrupt.zip.
    signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: sys.exit(143),
    )

    interrupted = False
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=(args.resume is None),
        )
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    finally:
        if interrupted:
            try:
                model.save(run_dir / "interrupt.zip")
            except Exception as exc:  # pragma: no cover
                print(f"Failed to save interrupt.zip: {exc}", file=sys.stderr)
        vec_env.close()

    if not interrupted:
        model.save(run_dir / "final.zip")


if __name__ == "__main__":
    main()
