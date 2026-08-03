#!/usr/bin/env python3
"""Golden-trajectory behaviour gate for DodgeHumanoid-v0.

Runs a fixed 300-step action sequence through a seed-0 env and hashes the
resulting (obs, reward, terminated, info["hits"]) stream with SHA-256.  Any
change to the environment's *computation* — physics substep interleaving, hit
registration, min-approach sampling, observation assembly, spawner RNG draw
order — moves the hash.  Pure refactors that preserve semantics do not.

The hash is bitwise: obs/reward are hashed as raw float64 little-endian bytes,
not as formatted text, so a 1-ULP drift is caught.

Usage::

    python bench/golden_traj.py            # print the hash
    python bench/golden_traj.py --check    # compare against bench/golden_traj.sha256

Exit code 0 = match (or plain print), 1 = mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import gymnasium as gym

import dodge_rl  # noqa: F401  registers DodgeHumanoid-v0

SEED = 0
N_STEPS = 300
HASH_FILE = Path(__file__).resolve().parent / "golden_traj.sha256"


def _action_sequence(action_space) -> np.ndarray:
    """Fixed action sequence, independent of the env's own RNG stream.

    Drawn from a dedicated Generator so the env's np_random (which drives
    spawning) is never advanced by the harness itself.
    """
    rng = np.random.default_rng(12345)
    low, high = action_space.low, action_space.high
    raw = rng.uniform(-1.0, 1.0, size=(N_STEPS, action_space.shape[0]))
    return np.clip(raw, low, high).astype(np.float64)


def trajectory_hash() -> str:
    env = gym.make("DodgeHumanoid-v0")
    try:
        obs, info = env.reset(seed=SEED)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(obs, dtype=np.float64).tobytes())
        h.update(np.int64(info["hits"]).tobytes())

        actions = _action_sequence(env.action_space)
        for i in range(N_STEPS):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            h.update(np.ascontiguousarray(obs, dtype=np.float64).tobytes())
            h.update(np.float64(reward).tobytes())
            h.update(np.int64(bool(terminated)).tobytes())
            h.update(np.int64(info["hits"]).tobytes())
            if terminated or truncated:
                # Deterministic restart so the stream stays fixed-length.
                obs, info = env.reset(seed=SEED + 1000 + i)
                h.update(np.ascontiguousarray(obs, dtype=np.float64).tobytes())
                h.update(np.int64(info["hits"]).tobytes())
        return h.hexdigest()
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the stored hash")
    parser.add_argument("--write", action="store_true", help="write the hash to the stored file")
    args = parser.parse_args()

    digest = trajectory_hash()
    print(digest)

    if args.write:
        HASH_FILE.write_text(digest + "\n")
        print(f"wrote {HASH_FILE}")
    if args.check:
        expected = HASH_FILE.read_text().strip()
        if digest != expected:
            print(f"MISMATCH: expected {expected}")
            return 1
        print("MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
