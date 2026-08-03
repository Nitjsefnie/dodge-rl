"""DodgeEnv: Gymnasium environment where a humanoid dodges projectiles.

The arena is ``assets/dodge_humanoid.xml`` (a 3x3 m lethal-walled cell; walls
are virtual — enforced by reward/termination, not geometry). Four projectile
bodies (``proj0..proj3``, freejoint spheres with ``gravcomp=1``) are parked
below the floor while inactive and launched at the humanoid by a stochastic
spawner.

Observation (float64, shape (73,)):
    humanoid qpos[2:] (root x,y excluded; 22) + full humanoid qvel (23)
    + 4 slot blocks of [active_flag, rel_pos(3), rel_vel(3)] (7 each),
    relative to the torso body, slots ordered by time-to-closest-approach
    ascending with inactive (all-zero) blocks last.

Reward per control step:
    +1.0 upright (torso z in [1.0, 2.0])
    -0.5 * sum over active projectiles of max(0, 1.2 - d)^2, d = distance to
        the nearest of {head, torso, pelvis}
    -50 per projectile/humanoid contact this step (projectile despawns, the
        episode continues)
    -100 and terminated when torso |x| > 1.5 or |y| > 1.5
    -0.05 * ||action||^2

info["min_approach"] uses -1.0 as a sentinel while no projectile has been
active yet this episode (a real approach distance is always >= 0).
"""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "dodge_humanoid.xml"

NUM_SLOTS = 4
FRAME_SKIP = 5  # control dt 0.015 s with physics dt 0.003 s

FIRST_SPAWN_TIME = 0.5  # seconds
SPAWN_INTERVAL_RANGE = (1.0, 2.5)  # seconds, uniform
SPAWN_RADIUS_RANGE = (8.0, 15.0)  # metres from the torso, uniform
SPAWN_AZIMUTH_RANGE = (0.0, 2.0 * np.pi)
SPAWN_ELEVATION_RANGE = (-10.0, 75.0)  # degrees, uniform
SPAWN_MIN_Z = 0.3
SPAWN_SPEED_RANGE = (4.0, 9.0)  # m/s, uniform
PRECISE_AIM_PROB = 0.5
AIM_OFFSET_STD = 0.35  # metres, per axis
TTL_FACTOR = 1.5  # TTL = TTL_FACTOR * initial_distance / speed
DESPAWN_DISTANCE = 20.0  # metres from the torso

TARGET_NAMES = ["head", "torso", "pelvis", "left_thigh", "right_thigh", "left_shin", "right_shin"]
TARGET_WEIGHTS = [0.15, 0.30, 0.20, 0.10, 0.10, 0.075, 0.075]
PROXIMITY_NAMES = ["head", "torso", "pelvis"]
PROXIMITY_RADIUS = 1.2

HIT_PENALTY = 50.0
WALL_PENALTY = 100.0
WALL_LIMIT = 1.5
UPRIGHT_Z_RANGE = (1.0, 2.0)
UPRIGHT_REWARD = 1.0
CTRL_COST_WEIGHT = 0.05
PROXIMITY_WEIGHT = 0.5
INIT_NOISE = 0.01  # uniform +- noise on humanoid qpos/qvel at reset


class DodgeEnv(MujocoEnv):
    """Projectile-dodging humanoid. See module docstring for the spec."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67,  # round(1 / dt), dt = 0.015 s
    }

    def __init__(self, xml_file=None, frame_skip=FRAME_SKIP, **kwargs):
        xml_file = xml_file or str(XML_PATH)
        # Resolve model-derived indices once from a probe model so the
        # observation space can be built before super().__init__ loads it
        # again; re-resolve against self.model afterwards (identical file).
        probe = mujoco.MjModel.from_xml_path(xml_file)
        obs_size = self._obs_size(probe)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        super().__init__(
            xml_file,
            frame_skip,
            observation_space,
            **kwargs,
        )
        self._resolve_indices()

        self._slots = [self._fresh_slot() for _ in range(NUM_SLOTS)]
        self._init_episode_state()

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_size(model):
        hum_nq = hum_nv = 0
        for jid in range(model.njnt):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[jid])
            if body_name.startswith("proj"):
                continue
            hum_nq += 7 if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            hum_nv += 6 if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
        return (hum_nq - 2) + hum_nv + NUM_SLOTS * 7

    def _resolve_indices(self):
        model = self.model

        torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._torso_bid = torso_bid

        # Humanoid subtree = torso body and all its descendants.
        hum_bodies = set()
        for bid in range(1, model.nbody):
            b = bid
            while b != 0:
                if b == torso_bid:
                    hum_bodies.add(bid)
                    break
                b = model.body_parentid[b]

        # Humanoid joint qpos/qvel indices, in joint order (never assume the
        # projectile freejoints come first/last — mask by joint address).
        hum_qpos_idx, hum_qvel_idx = [], []
        self._root_qposadr = self._root_dofadr = None
        for jid in range(model.njnt):
            if model.jnt_bodyid[jid] not in hum_bodies:
                continue
            is_free = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            qadr, vadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
            if is_free:
                self._root_qposadr, self._root_dofadr = qadr, vadr
            hum_qpos_idx.extend(range(qadr, qadr + (7 if is_free else 1)))
            hum_qvel_idx.extend(range(vadr, vadr + (6 if is_free else 1)))
        self._hum_qpos_idx = np.asarray(hum_qpos_idx)
        self._hum_qvel_idx = np.asarray(hum_qvel_idx)

        # Projectile slots: freejoint addresses, geom id, parked position.
        self._proj_qposadr = []
        self._proj_dofadr = []
        self._park_pos = []
        self._proj_geom_to_slot = {}
        for i in range(NUM_SLOTS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"proj{i}")
            jid = model.body_jntadr[bid]
            qadr, vadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
            self._proj_qposadr.append(qadr)
            self._proj_dofadr.append(vadr)
            self._park_pos.append(model.qpos0[qadr : qadr + 3].copy())
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"proj{i}_geom")
            self._proj_geom_to_slot[gid] = i

        # Humanoid geom ids (for hit detection).
        self._humanoid_geom_ids = frozenset(
            g for g in range(model.ngeom) if model.geom_bodyid[g] in hum_bodies
        )

        # Aim/proximity reference points: "head" is a geom, the rest bodies.
        self._point_refs = {}
        for name in set(TARGET_NAMES) | set(PROXIMITY_NAMES):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self._point_refs[name] = ("body", bid)
            else:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                assert gid != -1, f"no body or geom named {name}"
                self._point_refs[name] = ("geom", gid)

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_slot():
        return {"active": False, "spawn_time": 0.0, "ttl": 0.0}

    def _init_episode_state(self):
        self._next_spawn_time = FIRST_SPAWN_TIME
        self._step_hits = 0
        self._hits = 0
        self._spawns = 0
        self._wall_death = False
        self._min_approach = np.inf

    def _point_pos(self, name):
        kind, idx = self._point_refs[name]
        if kind == "body":
            return self.data.xpos[idx]
        return self.data.geom_xpos[idx]

    def _proj_pos(self, slot):
        qadr = self._proj_qposadr[slot]
        return self.data.qpos[qadr : qadr + 3]

    def _proj_vel(self, slot):
        vadr = self._proj_dofadr[slot]
        return self.data.qvel[vadr : vadr + 3]

    def _park(self, slot):
        """Deactivate a slot: back below the floor with zero velocity."""
        qadr = self._proj_qposadr[slot]
        vadr = self._proj_dofadr[slot]
        self.data.qpos[qadr : qadr + 3] = self._park_pos[slot]
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[vadr : vadr + 6] = 0.0
        self._slots[slot] = self._fresh_slot()

    # ------------------------------------------------------------------
    # Spawner
    # ------------------------------------------------------------------

    def _spawn_projectile(self, slot):
        """Launch one projectile from the given (free) slot. Returns the draw."""
        rng = self.np_random
        torso_pos = self.data.xpos[self._torso_bid]

        radius = rng.uniform(*SPAWN_RADIUS_RANGE)
        azimuth = rng.uniform(*SPAWN_AZIMUTH_RANGE)
        elevation = np.deg2rad(rng.uniform(*SPAWN_ELEVATION_RANGE))
        point = torso_pos + radius * np.array(
            [np.cos(elevation) * np.cos(azimuth),
             np.cos(elevation) * np.sin(azimuth),
             np.sin(elevation)]
        )
        if point[2] < SPAWN_MIN_Z:
            # Clamp z but keep the draw's radius: push the point out along its
            # azimuth so the torso distance stays exactly `radius`.
            dz = SPAWN_MIN_Z - torso_pos[2]
            horiz = np.sqrt(max(radius * radius - dz * dz, 0.0))
            point = np.array([
                torso_pos[0] + horiz * np.cos(azimuth),
                torso_pos[1] + horiz * np.sin(azimuth),
                SPAWN_MIN_Z,
            ])

        target = TARGET_NAMES[rng.choice(len(TARGET_NAMES), p=TARGET_WEIGHTS)]
        aim = self._point_pos(target).copy()
        if rng.random() >= PRECISE_AIM_PROB:
            aim = aim + rng.normal(0.0, AIM_OFFSET_STD, size=3)

        speed = rng.uniform(*SPAWN_SPEED_RANGE)
        direction = aim - point
        velocity = direction / np.linalg.norm(direction) * speed

        qadr = self._proj_qposadr[slot]
        vadr = self._proj_dofadr[slot]
        self.data.qpos[qadr : qadr + 3] = point
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[vadr : vadr + 3] = velocity
        self.data.qvel[vadr + 3 : vadr + 6] = 0.0

        distance = np.linalg.norm(point - torso_pos)
        self._slots[slot] = {
            "active": True,
            "spawn_time": self.data.time,
            "ttl": TTL_FACTOR * distance / speed,
        }
        self._spawns += 1
        return {"spawn_point": point, "aim_point": aim, "speed": speed}

    def _maybe_spawn(self):
        if self.data.time < self._next_spawn_time:
            return
        for slot in range(NUM_SLOTS):
            if not self._slots[slot]["active"]:
                self._spawn_projectile(slot)
                # The next interval is drawn when the spawn actually fires, so
                # a spawn deferred by full slots does not create a backlog.
                self._next_spawn_time = self.data.time + self.np_random.uniform(*SPAWN_INTERVAL_RANGE)
                return
        # All slots busy: the pending spawn stays pending for the next step.

    # ------------------------------------------------------------------
    # Hits, despawns
    # ------------------------------------------------------------------

    def _min_approach_distance(self, slot):
        pos = self._proj_pos(slot)
        return min(np.linalg.norm(pos - self._point_pos(n)) for n in PROXIMITY_NAMES)

    def _sample_min_approach(self):
        """Track the closest any active projectile got to head/torso/pelvis.

        Sampled after every physics substep so near-misses between control
        steps are not understated.
        """
        for slot in range(NUM_SLOTS):
            if self._slots[slot]["active"]:
                self._min_approach = min(self._min_approach, self._min_approach_distance(slot))

    def _register_hits(self):
        """Detect projectile/humanoid contacts after one physics substep."""
        if not any(s["active"] for s in self._slots):
            return
        hit_slots = set()
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            slot = self._proj_geom_to_slot.get(g1)
            other = g2
            if slot is None:
                slot = self._proj_geom_to_slot.get(g2)
                other = g1
            if slot is not None and other in self._humanoid_geom_ids:
                hit_slots.add(slot)
        for slot in hit_slots:
            if not self._slots[slot]["active"]:
                continue
            self._hits += 1
            self._step_hits += 1
            self._park(slot)

    def _despawn_expired(self, torso_pos):
        for slot in range(NUM_SLOTS):
            state = self._slots[slot]
            if not state["active"]:
                continue
            age = self.data.time - state["spawn_time"]
            if age > state["ttl"] or np.linalg.norm(self._proj_pos(slot) - torso_pos) > DESPAWN_DISTANCE:
                self._park(slot)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        data = self.data
        hum_qpos = data.qpos[self._hum_qpos_idx]
        hum_qvel = data.qvel[self._hum_qvel_idx]

        torso_pos = data.xpos[self._torso_bid]
        torso_vel = data.qvel[self._root_dofadr : self._root_dofadr + 3]

        active = [s for s in range(NUM_SLOTS) if self._slots[s]["active"]]

        def time_to_closest_approach(slot):
            rel = self._proj_pos(slot) - torso_pos
            rel_vel = self._proj_vel(slot) - torso_vel
            denom = rel_vel @ rel_vel
            if denom <= 0.0:
                return np.inf
            return max(0.0, -(rel @ rel_vel) / denom)

        active.sort(key=time_to_closest_approach)

        blocks = []
        for slot in active:
            rel = self._proj_pos(slot) - torso_pos
            rel_vel = self._proj_vel(slot) - torso_vel
            blocks.append(np.concatenate(([1.0], rel, rel_vel)))
        for _ in range(NUM_SLOTS - len(active)):
            blocks.append(np.zeros(7))

        return np.concatenate([hum_qpos[2:], hum_qvel] + blocks)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, "
                f"found {np.array(ctrl).shape}"
            )
        self.data.ctrl[:] = ctrl
        self._step_hits = 0
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
            self._sample_min_approach()
            self._register_hits()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.do_simulation(action, self.frame_skip)

        torso_pos = self.data.xpos[self._torso_bid].copy()

        self._despawn_expired(torso_pos)
        self._maybe_spawn()

        # Wall: lethal virtual walls at |x| = |y| = 1.5 m.
        terminated = bool(abs(torso_pos[0]) > WALL_LIMIT or abs(torso_pos[1]) > WALL_LIMIT)

        reward = 0.0
        if UPRIGHT_Z_RANGE[0] <= torso_pos[2] <= UPRIGHT_Z_RANGE[1]:
            reward += UPRIGHT_REWARD

        for slot in range(NUM_SLOTS):
            if not self._slots[slot]["active"]:
                continue
            d = self._min_approach_distance(slot)
            self._min_approach = min(self._min_approach, d)
            reward -= PROXIMITY_WEIGHT * max(0.0, PROXIMITY_RADIUS - d) ** 2

        reward -= HIT_PENALTY * self._step_hits
        if terminated:
            reward -= WALL_PENALTY
            self._wall_death = True
        reward -= CTRL_COST_WEIGHT * (action @ action)

        return self._get_obs(), float(reward), terminated, False, self._info()

    def _info(self):
        return {
            "hits": self._hits,
            "wall_death": self._wall_death,
            "spawns": self._spawns,
            # Sentinel: -1.0 while no projectile has been active this episode.
            "min_approach": self._min_approach if np.isfinite(self._min_approach) else -1.0,
        }

    def _get_reset_info(self):
        return self._info()

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[self._hum_qpos_idx] += self.np_random.uniform(
            -INIT_NOISE, INIT_NOISE, size=len(self._hum_qpos_idx)
        )
        qvel[self._hum_qvel_idx] += self.np_random.uniform(
            -INIT_NOISE, INIT_NOISE, size=len(self._hum_qvel_idx)
        )
        self.set_state(qpos, qvel)

        for slot in range(NUM_SLOTS):
            self._park(slot)
        self._init_episode_state()

        return self._get_obs()
