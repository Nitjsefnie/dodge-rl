"""Environment-level tests for DodgeEnv / DodgeHumanoid-v0.

Covers the Task 2 spec: registration + env_checker, spawn geometry, straight
flight, precise-aim geometry, wall termination, hit handling, determinism,
observation slot ordering, and an anatomy guard on the underlying model.

Fix round 1 added: aim-point pinning, exact reward decomposition, spawn
schedule bounds, despawn rules, per-substep hit detection, rel_vel frame
pinning, target-weight statistics, render metadata, and reset info keys.
"""

import math
import warnings
from collections import Counter
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
# humanoid qvel (23) -> 45, then the 4-element wall block, then 4 slot blocks
# of 7.
HUM_OBS_LEN = 45
WALL_OBS_LEN = 4
SLOT_BLOCK_LEN = 7
OBS_LEN = HUM_OBS_LEN + WALL_OBS_LEN + 4 * SLOT_BLOCK_LEN
SLOTS_START = HUM_OBS_LEN + WALL_OBS_LEN


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
    """The registered env passes gymnasium's compliance checker.

    skip_render_check=True: the render check instantiates every declared
    render_mode (including "human"), which aborts on this headless box
    (glfw get_video_mode). Rendering backends are Task 4's job.
    """
    env = gymnasium.make("DodgeHumanoid-v0")
    try:
        assert env.spec.max_episode_steps == 2000
        check_env(env.unwrapped, skip_render_check=True)
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

    # Aim pinning: a precise shot must aim exactly at the targeted link's
    # xpos, not at some self-reported point. _MidpointRng targets "torso".
    np.testing.assert_allclose(aim, _torso_pos(env), atol=1e-9)

    vadr = _freejoint_dofadr(env.model, "proj0")
    vel = env.data.qvel[vadr : vadr + 3].copy()

    t = np.dot(aim - spawn, vel) / np.dot(vel, vel)
    closest = spawn + t * vel
    assert np.linalg.norm(closest - aim) < 1e-6


def test_wall_termination(env):
    """Torso beyond |x| > 1.5 terminates, and the killing step scores 0.0."""
    _, reset_info = env.reset(seed=1)
    assert reset_info == {
        "hits": 0,
        "wall_death": False,
        "fall_death": False,
        "dodged": 0,
        "spawns": 0,
        "min_approach": -1.0,
    }
    qadr = _freejoint_qposadr(env.model, "torso")
    env.data.qpos[qadr] = 1.6
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    _, reward, terminated, truncated, info = env.step(np.zeros(env.model.nu))
    assert terminated
    assert not truncated
    assert info["wall_death"]
    # The killing step scores nothing; the shortened episode is the whole cost.
    assert reward == pytest.approx(0.0, abs=1e-12)


def _launch_through(env, start, velocity):
    """Activate slot 0 and override it onto a chosen straight-line course."""
    env._spawn_projectile(0)  # proper slot activation; placement overridden
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = np.asarray(start, dtype=np.float64)
    env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 3] = np.asarray(velocity, dtype=np.float64)
    env.data.qvel[vadr + 3 : vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_dodged_counts_only_projectiles_that_passed_through_the_cell(env):
    """A dodge is a shot that entered the 3x3x3 cell and left it again.

    Pins both halves: a shot crossing the cell counts once, and a shot that
    never enters counts zero. The distinction matters because "despawned
    without hitting" would also count shots that were never going to arrive,
    which would make the dodge rate look good for doing nothing.
    """
    # Crosses the cell, well above the humanoid so it cannot make contact.
    env.reset(seed=21)
    env.data.qvel[:] = 0.0
    torso_z = _torso_pos(env)[2]
    _launch_through(env, [0.5, 0.0, torso_z + 1.0], [10.0, 0.0, 0.0])
    info = {"dodged": 0}
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(np.zeros(env.model.nu))
        if info["dodged"] or terminated or truncated:
            break
    assert info["dodged"] == 1, "a shot crossing the cell must count as dodged"
    assert info["hits"] == 0, "this shot must not have touched the humanoid"

    # Never enters the cell: outside in x for its whole flight.
    env.reset(seed=22)
    env.data.qvel[:] = 0.0
    torso_z = _torso_pos(env)[2]
    _launch_through(env, [5.0, 0.0, torso_z + 1.0], [10.0, 0.0, 0.0])
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(np.zeros(env.model.nu))
        if terminated or truncated:
            break
    assert info["dodged"] == 0, "a shot that never entered the cell was not dodged"


def test_falling_is_lethal(env):
    """Torso below 0.5 m terminates, and the killing step scores 0.0.

    Falling is a death, not a penalty: that is what keeps survival time the
    only score while still making standing a precondition for it. Also pins
    that a crouch well above the limit does NOT kill, so the threshold cannot
    quietly drift upward into ordinary motion.
    """
    env.reset(seed=4)
    qadr = _freejoint_qposadr(env.model, "torso")

    # Fallen: lethal.
    env.data.qpos[qadr + 2] = 0.3
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert terminated, "a fallen humanoid must die"
    assert info["fall_death"]
    assert not info["wall_death"] and info["hits"] == 0, "wrong cause of death"
    assert reward == pytest.approx(0.0, abs=1e-12)

    # Upright: not lethal, and scores a normal surviving step.
    env.reset(seed=4)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert _torso_pos(env)[2] > 0.5, "fixture humanoid should start upright"
    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert not terminated and not info["fall_death"]
    assert reward == pytest.approx(1.0, abs=1e-12)


def test_hit_is_instantly_lethal_and_despawns(env):
    """A projectile forced into the torso: episode ends immediately, projectile parked.

    Any contact is fatal regardless of magnitude — there is no damage model and
    no partial hit, so this also pins that the episode does NOT continue.
    """
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
    assert terminated, "any projectile contact must end the episode"
    assert not env._slots[0]["active"], "hit projectile must despawn"
    # Parked back below the floor with zero velocity.
    assert env.data.qpos[qadr + 2] == pytest.approx(-10.0)
    assert np.all(env.data.qvel[vadr : vadr + 6] == 0.0)
    # The killing step scores nothing; the shortened episode is the whole cost.
    assert reward == pytest.approx(0.0, abs=1e-12)


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
    blocks = obs[SLOTS_START:].reshape(4, SLOT_BLOCK_LEN)

    assert blocks[0, 0] == 1.0 and blocks[1, 0] == 1.0
    np.testing.assert_allclose(blocks[0, 1:4], placements[1][0] - torso_pos, atol=1e-12)
    np.testing.assert_allclose(blocks[0, 4:7], placements[1][1], atol=1e-12)
    np.testing.assert_allclose(blocks[1, 1:4], placements[0][0] - torso_pos, atol=1e-12)
    np.testing.assert_allclose(blocks[1, 4:7], placements[0][1], atol=1e-12)
    assert np.all(blocks[2:] == 0.0), "inactive slots must be all-zero blocks"


def test_wall_block_reports_signed_distance_to_each_wall(env):
    """The wall block tracks torso x,y and crosses zero exactly at the boundary.

    Pins the sign convention, not just the arithmetic: an absolute-value
    encoding (1.5 - |x|) would pass a magnitude-only check while leaving the
    policy unable to tell which way to run, which is the whole point of the
    feature.
    """
    env.reset(seed=3)
    qadr = _freejoint_qposadr(env.model, "torso")

    for x, y in [(0.0, 0.0), (1.2, -0.4), (-1.45, 1.45)]:
        env.data.qpos[qadr] = x
        env.data.qpos[qadr + 1] = y
        mujoco.mj_forward(env.model, env.data)

        wall = env._get_obs()[HUM_OBS_LEN : HUM_OBS_LEN + WALL_OBS_LEN]
        tx, ty = _torso_pos(env)[:2]
        np.testing.assert_allclose(
            wall, [1.5 - tx, 1.5 + tx, 1.5 - ty, 1.5 + ty], atol=1e-12
        )
        assert np.all(wall > 0.0), "inside the arena every wall distance is positive"

    # Opposite walls must disagree once the torso is off-centre, which is what
    # distinguishes the signed encoding from the symmetric one.
    env.data.qpos[qadr] = 1.0
    env.data.qpos[qadr + 1] = 0.0
    mujoco.mj_forward(env.model, env.data)
    wall = env._get_obs()[HUM_OBS_LEN : HUM_OBS_LEN + WALL_OBS_LEN]
    assert wall[0] < wall[1], "+x wall must read nearer than -x wall at x = +1.0"

    # At the lethal boundary the corresponding entry is exactly zero: the sign
    # flip is the termination condition, expressed in the observation.
    env.data.qpos[qadr] = 1.5
    mujoco.mj_forward(env.model, env.data)
    wall = env._get_obs()[HUM_OBS_LEN : HUM_OBS_LEN + WALL_OBS_LEN]
    assert wall[0] == pytest.approx(0.0, abs=1e-12)


def test_anatomy_guard(env):
    """Joint ranges match stock values (env-level refactors can't swap models)."""
    model = env.model
    rk_id = _joint_id(model, "right_knee")
    assert model.jnt_range[rk_id][0] == pytest.approx(math.radians(-160), abs=1e-4)
    assert model.jnt_range[rk_id][1] == pytest.approx(math.radians(-2), abs=1e-4)

    # Shape guards tying the observation/action spaces to the stock humanoid.
    assert env.action_space.shape == (17,)
    assert env.observation_space.shape == (OBS_LEN,)


# ---------------------------------------------------------------------------
# Fix round 1: mutation-pinning and env-gap tests
# ---------------------------------------------------------------------------


def test_reward_is_exactly_survival_time(env):
    """A surviving step scores exactly 1.0 — nothing else may enter the reward.

    Deliberately arranged so that every term the reward used to carry would be
    non-zero if it came back: a projectile 0.6 m away (the old proximity
    shaping would contribute -0.18), a large non-zero action (the old control
    cost would contribute about -0.077), and a separate case with the torso
    outside the old upright band (which would cost the +1.0). Any of those
    resurfacing moves the reward off exactly 1.0, so this one assertion pins
    the absence of all three.
    """
    for z, label in ((None, "default pose"), (0.6, "below the old upright band")):
        env.reset(seed=11)
        env.data.qvel[:] = 0.0
        if z is not None:
            tadr = _freejoint_qposadr(env.model, "torso")
            env.data.qpos[tadr + 2] = z
        env._spawn_projectile(0)  # proper slot activation; placement overridden

        torso_pos = _torso_pos(env)
        qadr = _freejoint_qposadr(env.model, "proj0")
        vadr = _freejoint_dofadr(env.model, "proj0")
        env.data.qpos[qadr : qadr + 3] = torso_pos + np.array([0.6, 0.0, 0.0])
        env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[vadr : vadr + 6] = 0.0

        action = 0.3 * np.ones(env.model.nu)
        _, reward, terminated, _, info = env.step(action)
        assert not terminated, f"{label}: setup must not kill the humanoid"
        assert info["hits"] == 0
        assert reward == pytest.approx(1.0, abs=1e-12), (
            f"{label}: surviving step scored {reward}, expected exactly 1.0 — "
            "a shaping, upright or control-cost term has returned"
        )


def test_reward_is_zero_on_the_step_that_kills(env):
    """Both lethal endings score 0.0, so return equals steps survived exactly."""
    # Wall.
    env.reset(seed=1)
    qadr = _freejoint_qposadr(env.model, "torso")
    env.data.qpos[qadr] = 1.6
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert terminated and info["wall_death"]
    assert reward == pytest.approx(0.0, abs=1e-12)

    # Hit.
    env.reset(seed=2)
    env._spawn_projectile(0)
    torso_pos = _torso_pos(env)
    pqadr = _freejoint_qposadr(env.model, "proj0")
    pvadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[pqadr : pqadr + 3] = torso_pos
    env.data.qpos[pqadr + 3 : pqadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[pvadr : pvadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert terminated and info["hits"] == 1
    assert reward == pytest.approx(0.0, abs=1e-12)


def test_spawn_schedule(env, monkeypatch):
    """First spawn at t in (0.495, 0.515]; inter-spawn gaps within U(1.0, 2.5).

    Gaps are measured on the 0.015 s control grid, so the upper bound admits
    one control step of quantization; gaps spanning a full-slot stall are
    exempt from the upper bound (the spawn correctly stayed pending).
    """
    env.reset(seed=5)
    # The spawn timer is what is under test, not lethality. Now that any
    # contact ends the episode, an unlucky hit would cut the rollout short
    # after one or two spawns and the schedule would never be observed.
    # Suppressing hit registration isolates the timer without altering it.
    env._register_hits = lambda: None
    # Same reason for falling: under zero action the humanoid collapses within
    # a second and the episode would end long before the schedule is visible.
    # Disabling the floor rather than the timer keeps what is under test intact.
    monkeypatch.setattr("dodge_rl.dodge_env.FALL_LIMIT", -1.0)
    action = np.zeros(env.model.nu)
    spawn_times = []
    spawn_step_idx = []
    full_house_steps = set()
    prev_spawns = 0
    for step_i in range(700):  # 10.5 s
        _, _, terminated, _, info = env.step(action)
        if all(s["active"] for s in env._slots):
            full_house_steps.add(step_i)
        if info["spawns"] > prev_spawns:
            spawn_times.append(env.data.time)
            spawn_step_idx.append(step_i)
            prev_spawns = info["spawns"]
        if terminated or len(spawn_times) >= 6:
            break

    assert len(spawn_times) >= 4, f"only {len(spawn_times)} spawns observed"
    assert 0.495 < spawn_times[0] <= 0.515, f"first spawn at t={spawn_times[0]}"
    for i in range(1, len(spawn_times)):
        gap = spawn_times[i] - spawn_times[i - 1]
        assert gap >= 1.0 - 1e-9, f"gap {gap} below the 1.0 s draw floor"
        stalled = any(
            s in full_house_steps
            for s in range(spawn_step_idx[i - 1], spawn_step_idx[i] + 1)
        )
        if not stalled:
            assert gap <= 2.5 + 0.0151, f"gap {gap} above the 2.5 s draw ceiling"


def test_despawn_ttl(env):
    """A projectile is parked once its age exceeds 1.5 * distance / speed.

    Known geometry: 10 m from the torso, 5 m/s perpendicular to the torso
    direction -> ttl = 3.0 s, peak torso distance sqrt(100+225) = 18 m < 20,
    so only the TTL rule can fire.
    """
    env.reset(seed=13)
    env.data.qvel[:] = 0.0
    torso_pos = _torso_pos(env)
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = torso_pos + np.array([10.0, 0.0, 0.0])
    env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 3] = np.array([0.0, 5.0, 0.0])
    env.data.qvel[vadr + 3 : vadr + 6] = 0.0
    # Compose from the canonical slot shape rather than restating it: a
    # hand-built dict breaks (KeyError) every time the slot gains a field.
    env._slots[0] = {
        **env._fresh_slot(),
        "active": True,
        "spawn_time": env.data.time,
        "ttl": 1.5 * 10.0 / 5.0,
    }

    action = np.zeros(env.model.nu)
    t0 = env.data.time
    steps = 0
    while env._slots[0]["active"] and steps < 400:
        env.step(action)
        steps += 1
    assert not env._slots[0]["active"], "projectile never despawned"
    age = env.data.time - t0
    assert 3.0 < age <= 3.0 + 0.015 + 1e-9, f"despawn age {age}, ttl 3.0"


def test_ttl_derivation(env):
    """The env's own TTL derivation after a REAL spawn: 1.5 * dist / speed.

    Pins TTL_FACTOR: the hand-built slot in test_despawn_ttl bypasses
    _spawn_projectile, so only this assertion kills TTL_FACTOR mutations.
    """
    env.reset(seed=29)
    torso_pos = _torso_pos(env)
    info = env._spawn_projectile(0)
    expected = 1.5 * np.linalg.norm(info["spawn_point"] - torso_pos) / info["speed"]
    assert env._slots[0]["ttl"] == pytest.approx(expected, rel=1e-9)


def test_despawn_distance(env):
    """A projectile beyond 20 m from the torso is parked on the next step."""
    env.reset(seed=14)
    env._spawn_projectile(0)  # active slot; placement overridden below
    torso_pos = _torso_pos(env)
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = torso_pos + np.array([25.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 6] = 0.0

    env.step(np.zeros(env.model.nu))
    assert not env._slots[0]["active"], "projectile > 20 m away must despawn"
    assert env.data.qpos[qadr + 2] == pytest.approx(-10.0)  # parked


def test_hit_detected_mid_control_step(env):
    """A hit visible only in the early substeps must still be registered and kill.

    The projectile starts 0.06 m from the right_hand centre (surfaces
    overlap: 0.08 + 0.04 = 0.12) with a 9 m/s velocity carrying it away.
    Traced with this exact setup (seed 17): contacts are present in the
    checks after substeps 1 and 2 of the control step, and the pair is fully
    separated from substep 3 on (centre distance 0.141 m, i.e. a surface gap
    of ~0.021 m over the 0.08 + 0.04 radii, and growing) — so a hit check
    that only runs after the final substep would miss it entirely.
    """
    env.reset(seed=17)
    env.data.qvel[:] = 0.0
    env._spawn_projectile(0)  # active slot; placement overridden below

    hand_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand")
    hand_pos = env.data.geom_xpos[hand_gid].copy()
    out = hand_pos - _torso_pos(env)
    out /= np.linalg.norm(out)

    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = hand_pos + 0.06 * out
    env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 3] = 9.0 * out
    env.data.qvel[vadr + 3 : vadr + 6] = 0.0

    _, reward, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert info["hits"] == 1, "contact during early substeps was missed"
    assert terminated, "a glancing mid-substep contact is as lethal as any other"
    assert reward == pytest.approx(0.0, abs=1e-12)
    assert not env._slots[0]["active"]


def test_obs_rel_vel_is_torso_relative(env):
    """obs rel_vel == projectile velocity minus root translational velocity."""
    env.reset(seed=19)
    env.data.qvel[:] = 0.0
    root_vadr = _freejoint_dofadr(env.model, "torso")
    torso_vel = np.array([0.8, -0.5, 0.3])
    env.data.qvel[root_vadr : root_vadr + 3] = torso_vel

    env._spawn_projectile(0)  # active slot; placement overridden below
    torso_pos = _torso_pos(env)
    proj_vel = np.array([1.0, 2.0, 3.0])
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = torso_pos + np.array([4.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 3] = proj_vel

    obs = env._get_obs()
    block = obs[SLOTS_START : SLOTS_START + SLOT_BLOCK_LEN]
    assert block[0] == 1.0
    np.testing.assert_allclose(block[1:4], [4.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(block[4:7], proj_vel - torso_vel, atol=1e-12)


class _RecordingRng:
    """Wraps a Generator, recording every choice() draw; passes the rest through."""

    def __init__(self, gen):
        self._gen = gen
        self.choices = []

    def choice(self, a, size=None, replace=True, p=None):
        value = self._gen.choice(a, size=size, replace=replace, p=p)
        self.choices.append(int(value))
        return value

    def __getattr__(self, name):
        return getattr(self._gen, name)


def test_target_weights(env):
    """Empirical target frequencies match the spec weights (seeded).

    Spec order: [head, torso, pelvis, left_thigh, right_thigh, left_shin,
    right_shin] with weights [0.15, 0.30, 0.20, 0.10, 0.10, 0.075, 0.075].
    """
    env.reset(seed=23)
    recorder = _RecordingRng(env.np_random)
    env.np_random = recorder
    target_order = [
        "head", "torso", "pelvis",
        "left_thigh", "right_thigh", "left_shin", "right_shin",
    ]
    n = 800
    for _ in range(n):
        env._park(0)
        env._spawn_projectile(0)
    assert len(recorder.choices) == n
    counts = Counter(target_order[i] for i in recorder.choices)
    torso_freq = counts["torso"] / n
    head_freq = counts["head"] / n
    assert abs(torso_freq - 0.30) <= 0.08, f"torso frequency {torso_freq}"
    assert abs(head_freq - 0.15) <= 0.06, f"head frequency {head_freq}"


def test_render_metadata_declared():
    """gym.make(..., render_mode="rgb_array") constructs without render warnings.

    Does not call render(): rendering backends are Task 4's job.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gymnasium.make("DodgeHumanoid-v0", render_mode="rgb_array")
    try:
        render_warnings = [w for w in caught if "render" in str(w.message).lower()]
        assert not render_warnings, [str(w.message) for w in render_warnings]
        metadata = env.unwrapped.metadata
        assert set(metadata["render_modes"]) == {"human", "rgb_array", "depth_array"}
        assert metadata["render_fps"] == 67
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Fix round 2: remaining mutation kills
# ---------------------------------------------------------------------------


# test_upright_band_edges was removed here: the +1.0 upright term it pinned
# no longer exists. Survival time is the entire reward, so there is no
# upright band left to have edges. Its mutation-kill role is taken over by
# test_reward_is_exactly_survival_time, which fails if any pose-dependent
# term re-enters the reward.


class _ClampForcingRng(np.random.Generator):
    """Forced draws: r=8, azimuth=0, elevation=-10 deg, precise, speed 5.

    With the torso at z ~= 1.4 the raw spawn z is 1.4 + 8*sin(-10 deg) ~= 0.01,
    below the 0.3 floor, so the clamp branch must fire.
    """

    def __init__(self):
        super().__init__(np.random.PCG64(0))

    def uniform(self, low=0.0, high=1.0, size=None):
        assert size is None
        if low == 8.0 and high == 15.0:  # radius
            return 8.0
        if low == 0.0 and high == 2.0 * np.pi:  # azimuth
            return 0.0
        if low == -10.0 and high == 75.0:  # elevation
            return -10.0
        if low == 4.0 and high == 9.0:  # speed
            return 5.0
        raise AssertionError(f"unexpected uniform({low}, {high})")

    def random(self, size=None):
        assert size is None
        return 0.0  # precise aim, no offset draw

    def choice(self, a, size=None, replace=True, p=None):
        assert size is None
        return 1  # torso

    def normal(self, loc=0.0, scale=1.0, size=None):
        raise AssertionError("a precise shot must not draw an aim offset")


def test_spawn_z_clamp_preserves_radius(env):
    """The ratified z-clamp keeps the drawn radius, not a literal z overwrite.

    Forced draws put the raw spawn point at z ~= 0.01 (< 0.3), so the clamp
    fires. A literal z-clamp mutant leaves the torso distance at ~7.95 m and
    fails the radius assertion.
    """
    env.reset(seed=37)
    env.np_random = _ClampForcingRng()
    torso_pos = _torso_pos(env)
    drawn_radius = 8.0
    # Self-check that the forced draws really do exercise the clamp branch.
    assert torso_pos[2] + drawn_radius * math.sin(math.radians(-10.0)) < 0.3

    info = env._spawn_projectile(0)
    spawn = np.asarray(info["spawn_point"], dtype=np.float64)
    assert spawn[2] == pytest.approx(0.3, abs=1e-12)
    dist = np.linalg.norm(spawn - torso_pos)
    assert dist == pytest.approx(drawn_radius, rel=1e-9)


def test_min_approach_sampled_per_substep(env):
    """min_approach must reflect the closest approach mid-control-step.

    Fly-through at 30 m/s passing 0.3 m to the side of the PELVIS point
    (below arm reach; the nearest geoms — butt capsule, thigh tops — keep
    >= 0.04 m surface clearance along the whole path). Per substep the
    projectile moves 0.09 m; the closest sampled state is
    sqrt(0.045^2 + 0.3^2) ~= 0.3034 m, while an end-of-step-only sampler
    would see sqrt(0.225^2 + 0.3^2) = 0.375 m. Asserting < 0.34 separates
    the two.
    """
    env.reset(seed=41)
    env.data.qvel[:] = 0.0
    env._spawn_projectile(0)  # active slot; placement overridden below

    pelvis_pos = env.data.xpos[_body_id(env.model, "pelvis")].copy()
    qadr = _freejoint_qposadr(env.model, "proj0")
    vadr = _freejoint_dofadr(env.model, "proj0")
    env.data.qpos[qadr : qadr + 3] = pelvis_pos + np.array([-0.225, 0.3, 0.0])
    env.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[vadr : vadr + 3] = np.array([30.0, 0.0, 0.0])
    env.data.qvel[vadr + 3 : vadr + 6] = 0.0

    _, _, terminated, _, info = env.step(np.zeros(env.model.nu))
    assert not terminated
    assert info["hits"] == 0, "fly-through must not touch the humanoid"
    assert 0.29 < info["min_approach"] < 0.34, (
        f"min_approach {info['min_approach']}: per-substep sampling expects "
        "~0.303; an end-of-step-only sampler would report 0.375"
    )
