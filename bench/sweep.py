#!/usr/bin/env python3
"""Interleaved A/B sweep over threading configurations for PPO.learn throughput.

Each variant is a set of environment variables (torch intra-op threads, OpenMP
wait policy, ...) applied to an otherwise identical ``bench/bench_learn.py``
subprocess. Variants are run *interleaved*, one measurement each per round,
with the order rotated every round:

    round 0:  A B C
    round 1:  B C A
    round 2:  C A B

That matters more than it looks. Running all of A then all of B confounds the
variant with anything that drifts over the life of the machine — thermal
throttling, a noisy co-tenant, page-cache warmth. Interleaving spreads that
drift across every variant instead of loading it onto whichever ran last, and
rotating removes the fixed position-in-round bias.

Every measurement is a fresh process: the thread-count knobs are read by torch
and libgomp at start-up, so they cannot be A/B'd inside one interpreter.

Usage::

    python bench/sweep.py --rounds 5
    python bench/sweep.py --variants threads1,threads2 --rounds 7 --json-out r.json
    python bench/sweep.py --note "predict: threads=1 wins by >2x if spin-wait is real"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
BENCH_LEARN = _HERE / "bench_learn.py"

# Variant registry. The value is the environment overlay; an explicit empty
# string means "unset this variable for the child" (so a variant is not
# contaminated by whatever the surrounding shell exported).
VARIANTS: dict[str, dict[str, str]] = {
    # The hypothesis under test: torch/OpenMP parallel-region spin-wait on the
    # tiny per-step policy forwards. If it is real, dropping to a single
    # intra-op thread (no parallel region at all) or disabling the spin
    # (PASSIVE / zero blocktime) should move end-to-end throughput a lot.
    "threads1": {"DODGE_TORCH_THREADS": "1"},
    "threads2": {"DODGE_TORCH_THREADS": "2"},  # what training runs today
    "threads4": {"DODGE_TORCH_THREADS": "4"},
    "threads1-passive": {
        "DODGE_TORCH_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
    },
    "threads2-passive": {
        "DODGE_TORCH_THREADS": "2",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
    },
}

DEFAULT_VARIANTS = "threads1,threads2,threads1-passive,threads2-passive"

# Variables a variant may set; every one of them is cleared for variants that
# do not, so each child starts from the same baseline environment.
_MANAGED_VARS = (
    "DODGE_TORCH_THREADS",
    "OMP_WAIT_POLICY",
    "KMP_BLOCKTIME",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def child_env(overlay: dict[str, str]) -> dict[str, str]:
    """Full environment for one measurement: parent env, managed vars reset."""
    env = dict(os.environ)
    for var in _MANAGED_VARS:
        env.pop(var, None)
    env.update(overlay)
    return env


def measure(label: str, overlay: dict[str, str], total_timesteps: int, n_envs: int) -> dict:
    """One measurement of one variant, in a fresh process. Returns its JSON."""
    cmd = [
        sys.executable,
        str(BENCH_LEARN),
        "--json",
        "--repeats",
        "1",
        "--total-timesteps",
        str(total_timesteps),
        "--n-envs",
        str(n_envs),
        "--label",
        label,
    ]
    proc = subprocess.run(
        cmd,
        env=child_env(overlay),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"variant {label!r} failed (exit {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    # bench_learn.py prints its JSON as the last line.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"variant {label!r} produced no output\nstderr:\n{proc.stderr}")
    return json.loads(lines[-1])


def provenance() -> dict:
    """Machine facts a number is meaningless without."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # not linux
        affinity = None
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "loadavg": os.getloadavg(),
        "github_runner": os.environ.get("RUNNER_NAME"),
        "github_sha": os.environ.get("GITHUB_SHA"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception as exc:  # pragma: no cover - provenance only
        info["torch"] = f"unavailable: {exc}"
    return info


def summarize(rates: dict[str, list[float]], baseline: str) -> list[dict]:
    """Per-variant medians, sorted fastest first, relative to ``baseline``."""
    base_median = statistics.median(rates[baseline]) if rates.get(baseline) else None
    rows = []
    for label, values in rates.items():
        if not values:
            continue
        median = statistics.median(values)
        rows.append(
            {
                "variant": label,
                "n": len(values),
                "median_steps_per_s": round(median, 2),
                "min_steps_per_s": round(min(values), 2),
                "max_steps_per_s": round(max(values), 2),
                "spread_pct": round(100.0 * (max(values) - min(values)) / median, 1),
                "vs_baseline": round(median / base_median, 3) if base_median else None,
                "runs": [round(v, 2) for v in values],
            }
        )
    rows.sort(key=lambda r: r["median_steps_per_s"], reverse=True)
    return rows


def markdown_table(rows: list[dict], baseline: str) -> str:
    out = [
        f"| variant | n | median steps/s | min | max | spread | vs `{baseline}` |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        vs = f"{r['vs_baseline']:.2f}×" if r["vs_baseline"] is not None else "—"
        out.append(
            f"| `{r['variant']}` | {r['n']} | **{r['median_steps_per_s']}** | "
            f"{r['min_steps_per_s']} | {r['max_steps_per_s']} | {r['spread_pct']}% | {vs} |"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="comma-separated names from the registry")
    parser.add_argument("--rounds", type=int, default=5, help="measurements per variant (after warmup)")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="discarded leading rounds")
    parser.add_argument("--total-timesteps", type=int, default=3 * 6 * 2048)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--baseline", default="threads2", help="variant the ratios are relative to")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--note",
        default="",
        help="recorded verbatim in the result: use it for the PREDICTION, written before the run",
    )
    args = parser.parse_args()

    labels = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in labels if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}. known: {', '.join(VARIANTS)}")
    if len(labels) < 1:
        parser.error("need at least one variant")
    baseline = args.baseline if args.baseline in labels else labels[0]

    print(f"# sweep: {len(labels)} variants × {args.rounds} rounds "
          f"(+{args.warmup_rounds} warmup), {args.total_timesteps} timesteps each", flush=True)
    print(f"# provenance: {json.dumps(provenance())}", flush=True)
    if args.note:
        print(f"# note: {args.note}", flush=True)

    rates: dict[str, list[float]] = {label: [] for label in labels}
    total_rounds = args.warmup_rounds + args.rounds
    t_start = time.perf_counter()

    for rnd in range(total_rounds):
        warmup = rnd < args.warmup_rounds
        # Rotate so no variant sits permanently in the first (coldest) slot.
        shift = rnd % len(labels)
        order = labels[shift:] + labels[:shift]
        for label in order:
            result = measure(label, VARIANTS[label], args.total_timesteps, args.n_envs)
            rate = result["median_steps_per_s"]
            tag = "warmup" if warmup else f"round {rnd - args.warmup_rounds + 1}/{args.rounds}"
            print(f"  [{tag}] {label:<18} {rate:8.1f} steps/s", flush=True)
            if not warmup:
                rates[label].append(rate)

    elapsed = time.perf_counter() - t_start
    rows = summarize(rates, baseline)
    table = markdown_table(rows, baseline)

    print()
    print(table, flush=True)
    print(f"\nsweep wall-clock: {elapsed / 60:.1f} min", flush=True)

    payload = {
        "note": args.note,
        "baseline": baseline,
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "variant_env": {label: VARIANTS[label] for label in labels},
        "provenance": provenance(),
        "elapsed_s": round(elapsed, 1),
        "results": rows,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}", flush=True)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"## PPO learn() throughput sweep\n\n")
            if args.note:
                fh.write(f"**Prediction / note:** {args.note}\n\n")
            fh.write(
                f"{args.rounds} interleaved rounds per variant, {args.total_timesteps} timesteps, "
                f"{args.n_envs} envs, {payload['provenance']['cpu_count']} vCPU, "
                f"torch {payload['provenance']['torch']}.\n\n"
            )
            fh.write(table + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
