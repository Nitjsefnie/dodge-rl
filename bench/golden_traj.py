#!/usr/bin/env python3
"""Golden-trajectory behaviour gate for DodgeHumanoid-v0.

Runs a fixed 300-step action sequence through a seed-0 env and records the
resulting (obs, reward, terminated, info["hits"]) stream.  Any change to the
environment's *computation* — physics substep interleaving, hit registration,
min-approach sampling, observation assembly, spawner RNG draw order — moves the
trajectory.  Pure refactors that preserve semantics do not.

Two gates, because one of them turned out not to be portable:

``--check`` (bitwise, same machine)
    SHA-256 over the raw float64 little-endian bytes, so a 1-ULP drift is
    caught.  This is the strongest possible gate and the right one to run on
    the machine the reference hash was generated on.

``--check-approx`` (numeric, any machine)
    Compares against a stored reference trajectory within a tolerance.  The
    bitwise hash is **machine-dependent**: measured 2026-08-03 across six
    GitHub runners, it takes exactly one of two values, cd3999d6… on
    AMD EPYC 7763/9V74 (no avx512f) and 109b05a9… on Intel Xeon (avx512f),
    with the training box agreeing with the former.  Same code, same library
    versions, different silicon.  A bitwise gate in CI therefore fails on the
    luck of the runner draw, while a real semantic change moves the trajectory
    grossly and immediately — which is what the tolerance mode tests.

Usage::

    python bench/golden_traj.py                      # print the hash
    python bench/golden_traj.py --check              # bitwise, same machine
    python bench/golden_traj.py --dump ref.npz       # record a reference
    python bench/golden_traj.py --check-approx bench/golden_traj_ref.npz
    python bench/golden_traj.py --check-approx ... --profile   # divergence only

Exit code 0 = pass (or plain print), 1 = fail.
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
REF_FILE = Path(__file__).resolve().parent / "golden_traj_ref.npz"

# Pass criterion for --check-approx. PROVISIONAL: these are placeholders until
# the cross-population divergence profile has actually been measured. Run
# `--check-approx --profile` on both runner populations first, then set them
# from that data. Guessing a tolerance is how a gate ends up passing things it
# should catch.
DEFAULT_ATOL = 1e-9
DEFAULT_HORIZON = 25


def _action_sequence(action_space) -> np.ndarray:
    """Fixed action sequence, independent of the env's own RNG stream.

    Drawn from a dedicated Generator so the env's np_random (which drives
    spawning) is never advanced by the harness itself.
    """
    rng = np.random.default_rng(12345)
    low, high = action_space.low, action_space.high
    raw = rng.uniform(-1.0, 1.0, size=(N_STEPS, action_space.shape[0]))
    return np.clip(raw, low, high).astype(np.float64)


def trajectory_record() -> tuple[list[bytes], dict[str, np.ndarray]]:
    """Run the fixed sequence once.

    Returns the exact byte stream the hash is taken over, plus the same data
    as arrays for numeric comparison. The two are produced in one pass so they
    cannot describe different runs.
    """
    env = gym.make("DodgeHumanoid-v0")
    try:
        chunks: list[bytes] = []
        obs_seq: list[np.ndarray] = []
        rewards: list[float] = []
        terminations: list[int] = []
        hits: list[int] = []

        def emit_obs(value) -> None:
            arr = np.ascontiguousarray(value, dtype=np.float64)
            chunks.append(arr.tobytes())
            obs_seq.append(arr.copy())

        def emit_hits(value) -> None:
            scalar = np.int64(value)
            chunks.append(scalar.tobytes())
            hits.append(int(scalar))

        obs, info = env.reset(seed=SEED)
        emit_obs(obs)
        emit_hits(info["hits"])

        actions = _action_sequence(env.action_space)
        for i in range(N_STEPS):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            emit_obs(obs)
            reward_scalar = np.float64(reward)
            chunks.append(reward_scalar.tobytes())
            rewards.append(float(reward_scalar))
            term_scalar = np.int64(bool(terminated))
            chunks.append(term_scalar.tobytes())
            terminations.append(int(term_scalar))
            emit_hits(info["hits"])
            if terminated or truncated:
                # Deterministic restart so the stream stays fixed-length.
                obs, info = env.reset(seed=SEED + 1000 + i)
                emit_obs(obs)
                emit_hits(info["hits"])

        record = {
            "obs": np.stack(obs_seq),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "terminated": np.asarray(terminations, dtype=np.int64),
            "hits": np.asarray(hits, dtype=np.int64),
        }
        return chunks, record
    finally:
        env.close()


def trajectory_hash() -> str:
    chunks, _ = trajectory_record()
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def _obs_diff_profile(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Max absolute observation difference at each emission index."""
    return np.max(np.abs(current - reference), axis=1)


def compare_approx(record: dict[str, np.ndarray], ref_path: Path, atol: float,
                   horizon: int, profile_only: bool) -> int:
    ref = np.load(ref_path)

    # Structural divergence is not a tolerance question: a different number of
    # emissions means the episode terminated at different steps, which is a
    # behaviour change however small the float differences were.
    for key in ("obs", "rewards", "terminated", "hits"):
        if record[key].shape != ref[key].shape:
            print(f"STRUCTURAL MISMATCH in {key}: "
                  f"got {record[key].shape}, reference {ref[key].shape}")
            print("Different termination pattern — this is a behaviour change, not float drift.")
            return 1

    if not np.array_equal(record["terminated"], ref["terminated"]):
        idx = int(np.argmax(record["terminated"] != ref["terminated"]))
        print(f"TERMINATION MISMATCH: first differing step {idx}")
        return 1
    if not np.array_equal(record["hits"], ref["hits"]):
        idx = int(np.argmax(record["hits"] != ref["hits"]))
        print(f"HITS MISMATCH: first differing emission {idx}")
        return 1

    obs_profile = _obs_diff_profile(record["obs"], ref["obs"])
    reward_diff = np.abs(record["rewards"] - ref["rewards"])

    print(f"emissions: {obs_profile.size}  steps: {reward_diff.size}")
    print("max |obs diff| by emission index:")
    for idx in (0, 1, 2, 5, 10, 25, 50, 100, 200, obs_profile.size - 1):
        if idx < obs_profile.size:
            print(f"  [{idx:>4}] {obs_profile[idx]:.3e}")
    print(f"max |obs diff| overall : {obs_profile.max():.3e}")
    print(f"max |reward diff| overall: {reward_diff.max():.3e}")
    for threshold in (1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
        exceeded = np.nonzero(obs_profile > threshold)[0]
        first = int(exceeded[0]) if exceeded.size else None
        print(f"  first emission with |obs diff| > {threshold:.0e}: "
              f"{first if first is not None else 'never'}")

    if profile_only:
        return 0

    window = obs_profile[: horizon + 1]
    worst = float(window.max())
    if worst > atol:
        bad = int(np.argmax(window > atol))
        print(f"FAIL: |obs diff| {worst:.3e} exceeds atol {atol:.0e} "
              f"within the first {horizon} emissions (first at {bad})")
        return 1
    print(f"PASS: within atol {atol:.0e} over the first {horizon} emissions "
          f"(worst {worst:.3e})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="bitwise compare against the stored hash")
    parser.add_argument("--write", action="store_true", help="write the hash to the stored file")
    parser.add_argument("--dump", type=Path, default=None, help="write the reference trajectory npz")
    parser.add_argument("--check-approx", type=Path, nargs="?", const=REF_FILE, default=None,
                        metavar="REF_NPZ", help="numeric compare against a reference trajectory")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="emissions the tolerance is enforced over")
    parser.add_argument("--profile", action="store_true",
                        help="with --check-approx: report the divergence profile without failing")
    args = parser.parse_args()

    chunks, record = trajectory_record()
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    digest = h.hexdigest()
    print(digest)

    if args.dump:
        np.savez_compressed(args.dump, **record)
        print(f"wrote {args.dump}")
    if args.write:
        HASH_FILE.write_text(digest + "\n")
        print(f"wrote {HASH_FILE}")

    status = 0
    if args.check:
        expected = HASH_FILE.read_text().strip()
        if digest != expected:
            print(f"MISMATCH: expected {expected}")
            status = 1
        else:
            print("MATCH")
    if args.check_approx is not None:
        status |= compare_approx(record, args.check_approx, args.atol, args.horizon, args.profile)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
