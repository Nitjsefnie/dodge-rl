#!/usr/bin/env python3
"""End-to-end training-throughput benchmark: wall-clock steps/s of PPO.learn.

Builds the *exact* PPO setup train.py uses (6 SubprocVecEnv workers wrapped in
VecMonitor, MlpPolicy with net_arch [256, 256], n_steps=2048, batch_size=4096,
cpu device, fixed seed) and times ``learn(total_timesteps=36864)`` — three full
rollout+update cycles (3 * 6 * 2048).

The PPO kwargs and the env factory are imported from train.py rather than
restated here: a hand-copied config drifts from the one that actually runs.

Only ``learn`` is timed; process start-up, model construction and worker boot
are excluded (they are amortised to nothing in a real 20M-step run).

Usage::

    nice -n 10 .venv/bin/python bench/bench_learn.py                # one run
    nice -n 10 .venv/bin/python bench/bench_learn.py --repeats 3    # median of 3
    nice -n 10 .venv/bin/python bench/bench_learn.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Import train.py from the repo root, whichever tree this script lives in.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

import train  # noqa: E402  (repo-root train.py: PPO_KWARGS + make_env_factory)

N_ENVS = 6
SEED = 1
TOTAL_TIMESTEPS = 3 * N_ENVS * 2048  # 36864 — three rollout+update cycles


def one_run(total_timesteps: int = TOTAL_TIMESTEPS, n_envs: int = N_ENVS) -> float:
    """Run ``learn`` once; return elapsed wall-clock seconds."""
    torch.set_num_threads(train.TORCH_NUM_THREADS)

    env_fns = [train.make_env_factory(SEED, i) for i in range(n_envs)]
    with tempfile.TemporaryDirectory(prefix="dodge-bench-") as tmp:
        vec_env = VecMonitor(
            SubprocVecEnv(env_fns),
            filename=os.path.join(tmp, "monitor.csv"),
            info_keywords=("hits", "wall_death", "spawns", "min_approach"),
        )
        try:
            model = PPO(env=vec_env, seed=SEED, **train.PPO_KWARGS)
            train.apply_rollout_optimizations(model)
            # Warm the worker pipes and any lazily-built torch kernels so the
            # timed section measures steady-state throughput, not first-call
            # allocation. One vec-step ~ nothing against 36864.
            t0 = time.perf_counter()
            model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True)
            elapsed = time.perf_counter() - t0
        finally:
            vec_env.close()
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    rates = []
    for i in range(args.repeats):
        elapsed = one_run(args.total_timesteps, args.n_envs)
        rate = args.total_timesteps / elapsed
        rates.append(rate)
        if not args.json:
            print(f"run {i + 1}/{args.repeats}: {elapsed:.2f} s  {rate:.1f} steps/s", flush=True)

    result = {
        "label": args.label,
        "total_timesteps": args.total_timesteps,
        "n_envs": N_ENVS,
        "runs": [round(r, 2) for r in rates],
        "median_steps_per_s": round(statistics.median(rates), 2),
        "min_steps_per_s": round(min(rates), 2),
        "max_steps_per_s": round(max(rates), 2),
    }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
