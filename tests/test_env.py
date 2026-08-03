"""Environment-level tests for DodgeEnv / DodgeHumanoid-v0.

Covers the Task 2 spec: registration + env_checker, spawn geometry, straight
flight, precise-aim geometry, wall termination, hit handling, determinism,
observation slot ordering, and an anatomy guard on the underlying model.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import gymnasium
from gymnasium.utils.env_checker import check_env

import dodge_rl  # noqa: F401  (registers DodgeHumanoid-v0)
from dodge_rl.dodge_env import DodgeEnv

XML_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "dodge_humanoid.xml")

# Observation layout: humanoid qpos (24) minus root x,y -> 22, plus full
# humanoid qvel (23) -> 45, then 4 slot blocks of 7.
HUM_OBS_LEN = 45
SLOT_BLOCK_LEN = 7
OBS_LEN = HUM_OBS_LEN + 4 * SLOT_BLOCK_LEN


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _freejoint_qposadr(model, body_name):
    return model.jnt_qposadr[model.body_jntadr[_body_id(model, body_name)]]


def _freejoint_dofadr(model, body_name):
    return model.jnt_dofadr[model.body_jntadr[_body_id(model, body_name)]]


def _torso_pos(env):
    return env.data.xpos[_body_id(env.model, "torso")].copy()


@pytest.fixture
def env():
    e = DodgeEnv()
    yield e
    e.close()


def test_env_checker_passes():
    """The registered env passes gymnasium's compliance checker."""
    env = gymnasium.make("DodgeHumanoid-v0")
    try:
        assert env.spec.max_episode_steps == 2000
        check_env(env)
    finally:
        env.close()


def test_spawn_distribution(env):
    """100 spawns: distances in [8, 15] m from torso, z >= 0.3."""
    env.reset(seed=123)
    torso_pos = _torso_pos(env)
    qadr = _freejoint_qposadr(env.model, "proj0")
    for _ in range(100):
        env._park(0)
        env._spawn_projectile(0)
        pos = env.data.qpos[qadr : qadr + 3].copy()
        dist = np.linalg.norm(pos - torso_pos)
        assert 8.0 <= dist <= 15.0, f"spawn distance {dist} out of [8, 15]"
        assert pos[2] >= 0.3, f"spawn z {pos[2]} below 0.3"


def test_straight_flight(env):
    """An active projectile keeps constant velocity (< 1e-6 drift) for 20 steps."""
    env.reset(seed=7)
    env._spawn_projectile(0)
    vadr = _freejoint_dofadr(env.model, "proj0")
    v0 = env.data.qvel[vadr : vadr + 3].copy()
    assert np.linalg.norm(v0) > 1.0  # sanity: it was actually launched

    action = np.zeros(env.model.nu)
    for _ in range(20):
        env.step(action)
        assert env._slots[0]["active"], "projectile deactivated unexpectedly"

    v1 = env.data.qvel[vadr : vadr + 3].copy()
    assert np.linalg.norm(v1 - v0) < 1e-6, "projectile velocity drifted"


class _MidpointRng(np.random.Generator):
    """Deterministic RNG: midpoint uniforms, always-precise aim, torso target."""

    def __init__(self):
        super().__init__(np.random.PCG64(0))

    def uniform(self, low=0.0, high=1.0, size=None):
        assert size is None
        return (low + high) / 2.0

    def random(self, size=None):
        assert size is None
        return 0.0  # < 0.5 -> precise aim

    def choice(self, a, size=None, replace=True, p=None):
        assert size is None
        return 1  # index 1 == "torso" in the target list

    def normal(self, loc=0.0, scale=1.0, size=None):
        raise AssertionError("a precise shot must not draw an aim offset")


def test_precise_aim_geometry(env):
    """Precise shot: the (spawn, launch velocity) line passes through the aim point."""
    env.reset(seed=0)
    env.np_random = _MidpointRng()
    info = env._spawn_projectile(0)
    spawn = np.asarray(info["spawn_point"], dtype=np.float64)
    aim = np.asarray(info["aim_point"], dtype=np.float64)

    vadr = _freejoint_dofadr(env.model, "proj0")
    vel = env.data.qvel[vadr : vadr + 3].copy()

    t = np.dot(aim - spawn, vel) / np.dot(vel, vel)
    closest = spawn + t * vel
    assert np.linalg.norm(closest - aim) < 1e-6


def test_wall_termination(env):
    """Torso beyond |x| > 1.5 terminates with reward <= -100 + step bounds."""
    env.reset(seed=1)
    qadr = _freejoint_qposadr(env.model, "torso")
    env.data.qpos[qadr] = 1.6
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    _, reward, terminated, truncated, info = env.step(np.zeros(env.model.nu))
    assert terminated
    assert not truncated
    assert info["wall_death"]
    # reward = -100 + upright (0 or 1); no spawns yet at t = 0.015 s, zero action
    assert reward <= -99.0 + 1e-9
    assert reward >= -100.0 - 1e-9


def test_hit_penalty_and_despawn(env):
    """A projectile forced into the torso: -50 once, despawn, episode continues."""
    env.reset(seed=2)
    env._spawn_projectile(0)  # proper slot activation; placement overridden below

    torso_pos = _torso_pos(env)
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = torso_pos
    env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert info["hits"] == 1
    assert not terminated
    assert not env._slots[0]["active"], "hit projectile must despawn"
    # Parked back below the floor with zero velocity.
    assert env.data.qpos[qadr + 2] == pytest.approx(-10.0)
    assert np.all(env.data.qvel[vadr : vadr + 6] == 0.0)
    # reward = -50 + upright (0 or 1); no other active projectiles, zero action
    assert reward <= -49.0 + 1e-9
    assert reward >= -50.0 - 1e-9


def test_determinism():
    """Same seed + same 200-step action sequence -> identical obs and rewards."""
    env1, env2 = DodgeEnv(), DodgeEnv()
    try:
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.array_equal(obs1, obs2)

        rng = np.random.default_rng(0)
        actions = rng.uniform(-0.4, 0.4, size=(200, env1.model.nu))
        for action in actions:
            step1 = env1.step(action)
            step2 = env2.step(action)
            assert np.array_equal(step1[0], step2[0]), "observations diverged"
            assert step1[1] == step2[1], "rewards diverged"
            assert step1[2] == step2[2], "termination diverged"
            if step1[2]:
                break
    finally:
        env1.close()
        env2.close()


def test_slot_ordering(env):
    """Active slots are ordered by time-to-closest-approach; inactive trail as zeros."""
    env.reset(seed=3)
    env.data.qvel[:] = 0.0  # freeze: known velocities for t* computation
    env._spawn_projectile(0)
    env._spawn_projectile(1)

    torso_pos = _torso_pos(env)
    # slot 0: t* = 5 s; slot 1: t* = 3 s -> slot 1 must come first.
    placements = {
        0: (torso_pos + np.array([5.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
        1: (torso_pos + np.array([0.0, 3.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    }
    for slot, (pos, vel) in placements.items():
        qadr = _freejoint_qposadr(env.model, f"proj{slot}")
        vadr = _freejoint_dofadr(env.model, f"proj{slot}")
        env.data.qpos[qadr : qadr + 3] = pos
        env.data.qvel[vadr : vadr + 3] = vel

    obs = env._get_obs()
    assert obs.shape == (OBS_LEN,)
    blocks = obs[HUM_OBS_LEN:].reshape(4, SLOT_BLOCK_LEN)

    assert blocks[0, 0] == 1.0 and blocks[1, 0] == 1.0
    np.testing.assert_allclose(blocks[0, 1:4], placements[1][0] - torso_pos, atol=1e-12)
    np.testing.assert_allclose(blocks[0, 4:7], placements[1][1], atol=1e-12)
    np.testing.assert_allclose(blocks[1, 1:4], placements[0][0] - torso_pos, atol=1e-12)
    np.testing.assert_allclose(blocks[1, 4:7], placements[0][1], atol=1e-12)
    assert np.all(blocks[2:] == 0.0), "inactive slots must be all-zero blocks"


def test_anatomy_guard(env):
    """Joint ranges match stock values (env-level refactors can't swap models)."""
    model = env.model
    rk_id = _joint_id(model, "right_knee")
    assert model.jnt_range[rk_id][0] == pytest.approx(math.radians(-160), abs=1e-4)
    assert model.jnt_range[rk_id][1] == pytest.approx(math.radians(-2), abs=1e-4)

    # Shape guards tying the observation/action spaces to the stock humanoid.
    assert env.action_space.shape == (17,)
    assert env.observation_space.shape == (OBS_LEN,)
