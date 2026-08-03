"""Evaluate a policy on DodgeHumanoid-v0 and print aggregate statistics.

Usage:
    python eval.py --checkpoint runs/.../final.zip [--episodes 20] [--seed 0]
    python eval.py --random [--episodes 20] [--seed 0]

Runs episodes on a single (non-rendering) env. Checkpoint policies act with
deterministic=True; --random samples from the action space. Per-episode
metrics come from the final step's info dict. Exits 3 if --checkpoint points
at a file that does not exist.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to an SB3 PPO checkpoint (.zip).")
    policy.add_argument("--random", action="store_true",
                        help="Evaluate a uniform random policy.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.checkpoint is not None and not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 3

    import gymnasium

    import dodge_rl  # noqa: F401  (registers DodgeHumanoid-v0)

    model = None
    if args.checkpoint is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.checkpoint), device="cpu")

    env = gymnasium.make("DodgeHumanoid-v0")
    env.action_space.seed(args.seed)

    rewards, lengths, hits_list, wall_deaths, fall_deaths = [], [], [], [], []
    dodged_list: list[int] = []
    min_approaches, no_projectile = [], 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed if ep == 0 else None)
        total_reward, length = 0.0, 0
        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(length)
        hits_list.append(info["hits"])
        wall_deaths.append(bool(info["wall_death"]))
        fall_deaths.append(bool(info["fall_death"]))
        dodged_list.append(info["dodged"])
        if info["min_approach"] < 0.0:
            no_projectile += 1
        else:
            min_approaches.append(info["min_approach"])

    env.close()

    rewards = np.asarray(rewards)
    lengths = np.asarray(lengths, dtype=np.float64)
    hits_arr = np.asarray(hits_list, dtype=np.float64)
    wall_frac = float(np.mean(wall_deaths)) if wall_deaths else 0.0
    fall_frac = float(np.mean(fall_deaths)) if fall_deaths else 0.0
    dodged_arr = np.asarray(dodged_list, dtype=np.float64)
    # Only shots that actually reached the cell count as "arrived": a dodge
    # plus the hit that ended the episode. Spawns include shots still in
    # flight when the episode ended, which were neither dodged nor survived.
    arrived = float(dodged_arr.sum() + hits_arr.sum())
    dodge_rate = f"{dodged_arr.sum() / arrived:.3f}" if arrived > 0 else "n/a"
    min_app_str = f"{np.mean(min_approaches):.3f}" if min_approaches else "n/a"

    print(f"Policy:              {args.checkpoint if args.checkpoint else 'random'}")
    print(f"Episodes:            {args.episodes}  (seed {args.seed})")
    print(f"{'Metric':<30}{'Mean':>12}{'Std':>12}")
    print(f"{'episode reward':<30}{rewards.mean():>12.3f}{rewards.std():>12.3f}")
    print(f"{'episode length':<30}{lengths.mean():>12.1f}{lengths.std():>12.1f}")
    print(f"{'dodged / episode':<30}{dodged_arr.mean():>12.3f}{dodged_arr.std():>12.3f}")
    print(f"{'hits / episode':<30}{hits_arr.mean():>12.3f}{hits_arr.std():>12.3f}")
    print(f"{'dodge rate (dodged/arrived)':<30}{dodge_rate:>12}{'-':>12}")
    print(f"{'wall-death fraction':<30}{wall_frac:>12.3f}{'-':>12}")
    print(f"{'fall-death fraction':<30}{fall_frac:>12.3f}{'-':>12}")
    print(f"{'min_approach (projectile eps)':<30}{min_app_str:>12}{'-':>12}")
    print(f"{'episodes with no projectile':<30}{no_projectile:>12d}{'-':>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
