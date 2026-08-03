"""Model-level tests for the dodge-humanoid MuJoCo asset."""

import math
import numpy as np
import mujoco
import pytest

XML_PATH = "assets/dodge_humanoid.xml"


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(XML_PATH)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _freejoint_qposadr(model, body_name):
    """Return qpos address for the body's free joint."""
    bid = _body_id(model, body_name)
    jid = model.body_jntadr[bid]
    return model.jnt_qposadr[jid]


def _freejoint_dofadr(model, body_name):
    """Return dof address for the body's free joint."""
    bid = _body_id(model, body_name)
    jid = model.body_jntadr[bid]
    return model.jnt_dofadr[jid]


def test_model_loads_and_has_projectiles(model):
    """The compiled model includes exactly four projectile bodies."""
    assert model.nbody == 18  # original 14 + 4 projectiles
    for i in range(4):
        body_name = f"proj{i}"
        bid = _body_id(model, body_name)
        assert bid != -1, f"missing body {body_name}"
        assert model.body_gravcomp[bid] == 1, f"{body_name} gravcomp != 1"

        jnt_name = f"{body_name}_freejoint"
        jid = _joint_id(model, jnt_name)
        assert jid != -1, f"missing joint {jnt_name}"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE


def test_joint_ranges_unchanged(model):
    """Humanoid joint limits are exactly the stock Gymnasium values."""
    # right_knee: -160 to -2 degrees
    rk_id = _joint_id(model, "right_knee")
    assert model.jnt_range[rk_id][0] == pytest.approx(math.radians(-160), abs=1e-4)
    assert model.jnt_range[rk_id][1] == pytest.approx(math.radians(-2), abs=1e-4)

    # left_elbow: -90 to 50 degrees
    le_id = _joint_id(model, "left_elbow")
    assert model.jnt_range[le_id][0] == pytest.approx(math.radians(-90), abs=1e-4)
    assert model.jnt_range[le_id][1] == pytest.approx(math.radians(50), abs=1e-4)


def test_projectile_parked_below_floor(data):
    """Parked projectiles sit below the floor plane and are inert."""
    for i in range(4):
        expected_z = -10.0 - i
        qadr = _freejoint_qposadr(data.model, f"proj{i}")
        assert data.qpos[qadr] == pytest.approx(0.0)
        assert data.qpos[qadr + 1] == pytest.approx(0.0)
        assert data.qpos[qadr + 2] == pytest.approx(expected_z)


def _set_freejoint_pos(data, body_name, pos):
    qadr = _freejoint_qposadr(data.model, body_name)
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])


def _set_freejoint_vel(data, body_name, vel):
    vadr = _freejoint_dofadr(data.model, body_name)
    data.qvel[vadr : vadr + 3] = vel
    data.qvel[vadr + 3 : vadr + 6] = 0.0


def _contacts_involving_geom(data, geom_id):
    """Return list of (geom1_id, geom2_id) for active contacts involving geom_id."""
    pairs = []
    for c in data.contact[: data.ncon]:
        if c.geom1 == geom_id or c.geom2 == geom_id:
            pairs.append((c.geom1, c.geom2))
    return pairs


def test_projectile_does_not_collide_with_floor(model, data):
    """Projectile with contype 2 / conaffinity 0 passes through the floor."""
    proj0_geom = _geom_id(model, "proj0_geom")
    _set_freejoint_pos(data, "proj0", np.array([2.5, 2.5, 0.05]))
    _set_freejoint_vel(data, "proj0", np.array([1.0, -0.5, 2.0]))

    mujoco.mj_forward(model, data)
    dofadr = _freejoint_dofadr(model, "proj0")
    start_vel = data.qvel[dofadr : dofadr + 6].copy()

    for _ in range(20):
        mujoco.mj_step(model, data)

    contacts = _contacts_involving_geom(data, proj0_geom)
    assert len(contacts) == 0, f"unexpected contacts involving proj0: {contacts}"

    end_vel = data.qvel[dofadr : dofadr + 6].copy()
    assert np.linalg.norm(start_vel - end_vel) < 1e-6, "projectile velocity changed"


def test_projectile_collides_with_humanoid(model, data):
    """Projectile collides with humanoid geoms when overlapped."""
    proj0_geom = _geom_id(model, "proj0_geom")
    # Place proj0 inside the torso1 capsule (center of torso).
    _set_freejoint_pos(data, "proj0", np.array([0.0, 0.0, 1.4]))
    _set_freejoint_vel(data, "proj0", np.array([0.0, 0.0, 0.0]))

    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    contacts = _contacts_involving_geom(data, proj0_geom)
    assert len(contacts) > 0, "expected proj0 to contact the humanoid"

    humanoid_geom_names = [
        "torso1", "head", "uwaist", "lwaist", "butt",
        "right_thigh1", "right_shin1", "right_foot",
        "left_thigh1", "left_shin1", "left_foot",
        "right_uarm1", "right_larm", "right_hand",
        "left_uarm1", "left_larm", "left_hand",
    ]
    humanoid_geom_ids = {_geom_id(model, n) for n in humanoid_geom_names}
    g1, g2 = contacts[0]
    other = g2 if g1 == proj0_geom else g1
    assert other in humanoid_geom_ids, f"proj0 contacted non-humanoid geom id {other}"


def test_projectile_straight_flight(model, data):
    """With gravcomp=1 and no contact, a projectile maintains its velocity."""
    v = np.array([3.0, -2.0, 1.5])
    _set_freejoint_pos(data, "proj0", np.array([0.0, 0.0, 2.0]))
    _set_freejoint_vel(data, "proj0", v)

    mujoco.mj_forward(model, data)

    dofadr = _freejoint_dofadr(model, "proj0")
    for _ in range(50):
        mujoco.mj_step(model, data)
        current_v = data.qvel[dofadr : dofadr + 3]
        assert np.linalg.norm(current_v - v) < 1e-6
