"""DodgeEnv: Gymnasium environment where a humanoid dodges projectiles.

The arena is ``assets/dodge_humanoid.xml`` (a 3x3 m lethal-walled cell; walls
are virtual — enforced by reward/termination, not geometry). Four projectile
bodies (``proj0..proj3``, freejoint spheres with ``gravcomp=1``) are parked
below the floor while inactive and launched at the humanoid by a stochastic
spawner.

Observation (float64, shape (77,)):
    humanoid qpos[2:] (root x,y excluded; 22) + full humanoid qvel (23)
    + wall block [1.5-x, 1.5+x, 1.5-y, 1.5+y] from torso xpos (4)
    + 4 slot blocks of [active_flag, rel_pos(3), rel_vel(3)] (7 each),
    relative to the torso body, slots ordered by time-to-closest-approach
    ascending with inactive (all-zero) blocks last.

Reward per control step:
    +1.0 for every step survived, 0.0 on the step that ends the episode.

    That is the whole reward. There is no shaping, no upright bonus, no
    control cost and no penalty term: the episode ending early IS the
    punishment, and total return equals steps survived. Three failure modes
    terminate — ANY projectile contact however glancing, torso
    |x| > 1.5 or |y| > 1.5, and torso z < 0.5 (fallen).

info["min_approach"] is diagnostics only — it no longer enters the reward,
but it is the metric that shows whether near-misses are tightening.

info["min_approach"] uses -1.0 as a sentinel while no projectile has been
active yet this episode (a real approach distance is always >= 0).
"""

import os
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

WALL_LIMIT = 1.5

# Torso height below which the humanoid counts as fallen, and dies.
#
# Measured, not guessed: with survival-only reward and no upright term, the
# humanoid spent 83-84% of every episode with its torso below 0.5 m (standing
# is ~1.3-1.4 m) and a policy trained for 1M steps was statistically identical
# to random — 283.2 vs 283.6 mean episode length over 150 matched episodes.
# A sprawled humanoid cannot dodge, so episode length was decided by when a
# projectile happened to arrive rather than by anything the policy did, and
# the gradient drowned in spawn noise. The hit distribution said the same
# thing: butt 20.6%, feet 11.1% — the contact profile of a body lying down.
#
# Making falling lethal keeps the score purely survival time (no upright
# bonus, no shaping) while making standing instrumentally necessary, and
# collapses the credit assignment to a single step, exactly as a hit does.
# 0.5 m sits well below a deep crouch and well above lying down.
FALL_LIMIT = 0.5

# Height of the cell the humanoid is confined to. Only used to decide whether
# a projectile is "inside the box" for the dodge counter — the ceiling is not
# lethal and has no geometry, exactly like the walls.
ARENA_HEIGHT = 3.0

# Diagnostic: when DODGE_DEBUG_HIT_GEOMS=1, info gains "hit_geoms", a per-geom
# tally of what the projectiles actually struck. Off by default and inert when
# off, so it costs nothing and changes no observation, reward or hash.
#
# Why it exists: hit detection fires on EVERY humanoid geom, while
# min_approach is measured only against PROXIMITY_NAMES (head, torso, pelvis).
# With ~0.98 hits per episode but a mean min_approach of ~0.32 — far above the
# ~0.15 m contact threshold — most contacts must be landing on geoms the
# metric never watches. This says which.
DEBUG_HIT_GEOMS = os.environ.get("DODGE_DEBUG_HIT_GEOMS") == "1"

# Survival time is the entire score: +1 per step survived, 0 on the step that
# kills. Total episode return is therefore exactly the number of steps the
# humanoid stayed alive, which is the quantity being optimised — no shaping
# term can trade against it or be gamed.
SURVIVAL_REWARD = 1.0

# Observation block reporting distance to each of the four virtual walls, in
# the order (+x, -x, +y, -y). Root x,y are excluded from the humanoid section
# by design, which left the arena — the only terminal failure mode — entirely
# invisible to the policy while it was ending 24-33% of episodes.
WALL_OBS_LEN = 4
INIT_NOISE = 0.01  # uniform +- noise on humanoid qpos/qvel at reset


class DodgeEnv(MujocoEnv):
    """Projectile-dodging humanoid. See module docstring for the spec."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67,  # round(1 / dt), dt = 0.015 s
    }

    def __init__(self, xml_file=None, frame_skip=FRAME_SKIP, **kwargs):
        xml_file = xml_file or str(XML_PATH)
        # Load the model once: the observation space is sized from it before
        # super().__init__, which then adopts the same model through the
        # _initialize_simulation override instead of re-parsing the XML.
        self._probe_model = mujoco.MjModel.from_xml_path(xml_file)
        obs_size = self._obs_size(self._probe_model)
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

    def _initialize_simulation(self):
        """Adopt the model loaded in __init__ instead of re-loading the file."""
        model = self._probe_model
        del self._probe_model
        # Same offscreen buffer sizing as the stock MujocoEnv loader.
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        return model, mujoco.MjData(model)

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _humanoid_body_ids(model):
        """Return (torso body id, set of humanoid body ids).

        The humanoid subtree is the torso body and all its descendants;
        projectiles are excluded by tree ancestry, not by name.
        """
        torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        hum_bodies = set()
        for bid in range(1, model.nbody):
            b = bid
            while b != 0:
                if b == torso_bid:
                    hum_bodies.add(bid)
                    break
                b = model.body_parentid[b]
        return torso_bid, hum_bodies

    @classmethod
    def _obs_size(cls, model):
        _, hum_bodies = cls._humanoid_body_ids(model)
        hum_nq = hum_nv = 0
        for jid in range(model.njnt):
            if model.jnt_bodyid[jid] not in hum_bodies:
                continue
            is_free = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            hum_nq += 7 if is_free else 1
            hum_nv += 6 if is_free else 1
        return (hum_nq - 2) + hum_nv + WALL_OBS_LEN + NUM_SLOTS * 7

    def _resolve_indices(self):
        model = self.model

        torso_bid, hum_bodies = self._humanoid_body_ids(model)
        self._torso_bid = torso_bid

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
        # The obs drops root x,y by ADDRESS via _root_qposadr — the root
        # freejoint is not assumed to be the first humanoid joint.
        self._hum_qpos_obs_mask = ~np.isin(
            self._hum_qpos_idx, (self._root_qposadr, self._root_qposadr + 1)
        )

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
        return {"active": False, "spawn_time": 0.0, "ttl": 0.0, "in_box": False}

    def _init_episode_state(self):
        self._next_spawn_time = FIRST_SPAWN_TIME
        self._step_hits = 0
        self._hit_geoms = {}
        self._hits = 0
        self._spawns = 0
        self._wall_death = False
        self._fall_death = False
        self._dodged = 0
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
            "in_box": False,
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
                if DEBUG_HIT_GEOMS:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other)
                    )
                    key = name or f"geom{int(other)}"
                    self._hit_geoms[key] = self._hit_geoms.get(key, 0) + 1
        for slot in hit_slots:
            if not self._slots[slot]["active"]:
                continue
            self._hits += 1
            self._step_hits += 1
            self._park(slot)

    def _update_dodges(self):
        """Count projectiles that entered the arena cell and left it again.

        A dodge is a projectile that passed *all the way through* the 3x3x3
        cell the humanoid is confined to without touching it — it must have
        been inside and then left. Deliberately not "it despawned without
        hitting": a shot that expired on TTL while still inside, or one that
        was always going to miss the cell entirely, was never dodged. A
        projectile that hit is parked during the physics substeps, before this
        runs, so it cannot be counted here.
        """
        for slot in range(NUM_SLOTS):
            state = self._slots[slot]
            if not state["active"]:
                continue
            pos = self._proj_pos(slot)
            inside = (
                abs(pos[0]) <= WALL_LIMIT
                and abs(pos[1]) <= WALL_LIMIT
                and 0.0 <= pos[2] <= ARENA_HEIGHT
            )
            if inside:
                state["in_box"] = True
            elif state["in_box"]:
                state["in_box"] = False
                self._dodged += 1

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
        hum_qpos = data.qpos[self._hum_qpos_idx][self._hum_qpos_obs_mask]
        hum_qvel = data.qvel[self._hum_qvel_idx]

        torso_pos = data.xpos[self._torso_bid]
        torso_vel = data.qvel[self._root_dofadr : self._root_dofadr + 3]

        active = [s for s in range(NUM_SLOTS) if self._slots[s]["active"]]
        rels = {
            s: (self._proj_pos(s) - torso_pos, self._proj_vel(s) - torso_vel)
            for s in active
        }

        def time_to_closest_approach(slot):
            rel, rel_vel = rels[slot]
            denom = rel_vel @ rel_vel
            if denom <= 0.0:
                return np.inf
            return max(0.0, -(rel @ rel_vel) / denom)

        active.sort(key=time_to_closest_approach)

        # Signed distance to each wall, not WALL_LIMIT - |x|. The absolute-value
        # form is symmetric in x, so it tells the policy how close the wall is
        # but not which way to run from it — it would have to infer the sign
        # from several steps of motion, which is the blindness this is meant to
        # remove. Each entry is positive inside the arena and crosses zero
        # exactly where the episode terminates.
        wall = np.array([
            WALL_LIMIT - torso_pos[0],
            WALL_LIMIT + torso_pos[0],
            WALL_LIMIT - torso_pos[1],
            WALL_LIMIT + torso_pos[1],
        ])

        blocks = []
        for slot in active:
            rel, rel_vel = rels[slot]
            blocks.append(np.concatenate(([1.0], rel, rel_vel)))
        for _ in range(NUM_SLOTS - len(active)):
            blocks.append(np.zeros(7))

        return np.concatenate([hum_qpos, hum_qvel, wall] + blocks)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # The parent do_simulation keeps its action-shape validation; only
        # the substep loop is overridden so hit detection and min_approach
        # sampling interleave with every physics substep.
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
            self._sample_min_approach()
            self._register_hits()
        # Populates cfrc_ext et al., as the stock implementation does.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self._step_hits = 0
        self.do_simulation(action, self.frame_skip)

        torso_pos = self.data.xpos[self._torso_bid].copy()

        # Before despawn, so a projectile that leaves the cell and is then
        # retired by TTL or distance still registers as having passed through.
        self._update_dodges()
        self._despawn_expired(torso_pos)
        self._maybe_spawn()

        # Diagnostics only: min_approach no longer shapes the reward, but it is
        # the metric that shows whether near-misses are tightening over
        # training. Kept inside the loop over ACTIVE slots so a despawned
        # projectile cannot keep contributing.
        for slot in range(NUM_SLOTS):
            if not self._slots[slot]["active"]:
                continue
            self._min_approach = min(self._min_approach, self._min_approach_distance(slot))

        # Three lethal failure modes, all terminal.
        wall_out = bool(abs(torso_pos[0]) > WALL_LIMIT or abs(torso_pos[1]) > WALL_LIMIT)
        if wall_out:
            self._wall_death = True
        # ANY contact kills, however glancing: there is no partial hit and no
        # damage model, so a graze is exactly as fatal as a direct hit.
        hit = self._step_hits > 0
        # Falling kills too. Not a penalty term — a death — so survival time
        # stays the only score while standing becomes a precondition for it.
        fallen = bool(torso_pos[2] < FALL_LIMIT)
        if fallen:
            self._fall_death = True
        terminated = bool(wall_out or hit or fallen)

        # Survival time is the score. No shaping term, no penalty: the episode
        # ending early is itself the entire cost, and undiscounted return
        # equals the number of steps survived.
        reward = 0.0 if terminated else SURVIVAL_REWARD

        return self._get_obs(), float(reward), terminated, False, self._info()

    def _info(self):
        info = {
            "hits": self._hits,
            "wall_death": self._wall_death,
            "fall_death": self._fall_death,
            # Projectiles that passed all the way through the cell: the count
            # of shots actually evaded, as opposed to shots that were never
            # going to arrive.
            "dodged": self._dodged,
            "spawns": self._spawns,
            # Sentinel: -1.0 while no projectile has been active this episode.
            "min_approach": self._min_approach if np.isfinite(self._min_approach) else -1.0,
        }
        if DEBUG_HIT_GEOMS:
            # Only present when the flag is set, so the default info contract
            # (and every test asserting it exactly) is untouched.
            info["hit_geoms"] = dict(self._hit_geoms)
        return info

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
