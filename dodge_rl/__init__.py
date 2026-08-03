"""dodge_rl: PPO humanoid projectile-dodging environments."""

from gymnasium.envs.registration import register

register(
    id="DodgeHumanoid-v0",
    entry_point="dodge_rl.dodge_env:DodgeEnv",
    max_episode_steps=2000,
)
