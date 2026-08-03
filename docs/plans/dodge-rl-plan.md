# Dodge-RL — humanoid projectile-dodging, implementation plan

Train a PPO policy giving full-body torque control of the MuJoCo humanoid so it
dodges slow-ish projectiles. Feasibility was measured on this box 2026-08-03:
Humanoid-v5 steps at ~1.6k/s per process, ~8.8k/s aggregate over 6 processes;
all-in PPO throughput ~2-3.5k env-steps/s; 20-100M-step runs fit in 1.5-15 h.
Reference prior art: PAC-MAN (arXiv:2607.28623) — same task, PPO, closest-approach
evasion reward + upright terms + fall penalty.

## Context

- Box: 12 vCPU EPYC Zen1, no GPU, 47 GiB RAM, shared with other tenants.
- Python 3.13.14 system; torch 2.13.0+cpu already installed system-wide.
- Repo: /root/dodge-rl, branch `main` (SDD serializes onto main; no worktrees).

## Global Constraints

- Venv at `/root/dodge-rl/.venv` created with `--system-site-packages` (reuses
  system torch 2.13.0+cpu — never install torch in the venv, never pip install
  into the system python). All python invocations in this repo go through
  `.venv/bin/python`.
- Every heavy process (pip resolving, tests that step the sim a lot, training,
  rendering) runs under `nice -n 10`. Training uses at most 6 worker processes.
- Design invariants (user-set, binding):
  1. **Reactive, not memorized** — every projectile's spawn position, direction,
     speed, timing and target are drawn from the env's seeded RNG each episode;
     no fixed scenarios anywhere.
  2. **Joint anatomy respected** — the stock Gymnasium humanoid joint `range`
     values are used UNCHANGED. Never widen or remove a joint range or actuator
     `ctrlrange`.
  3. **Arena 3×3×3 m** — humanoid torso crossing |x|>1.5 or |y|>1.5 (metres)
     terminates the episode with the wall penalty. The floor never terminates.
     The ceiling never terminates. There is NO wall geometry — boundary is a
     position check only.
  4. **Projectiles**: spawn 8-15 m from the torso (outside the arena), fly in a
     straight line at constant velocity (gravity-compensated body), aim computed
     ONCE at launch (no homing), pass through the arena boundary and the floor
     (contact only with humanoid geoms), despawn on hit, on TTL expiry, or when
     >20 m from the torso.
  5. **Aiming**: target is a sampled humanoid body link (so a prone body gets
     targeted where it actually is, incl. shots from above); a coin decides
     precise aim at the link centre vs. aim with Gaussian offset.
- Determinism: same `reset(seed=N)` + same action sequence ⇒ identical
  observation trajectory. All randomness flows through `self.np_random`.
- Tests are pytest, live in `tests/`, run via `nice -n 10 .venv/bin/python -m
  pytest`. A test encodes the spec: never relax a test to make an
  implementation pass.
- Commits: one logical change per commit, explicit paths staged (never
  `git add -A`), `Co-Authored-By` trailer naming the implementing model.
- No GPU code paths anywhere. Rendering uses `MUJOCO_GL=osmesa` (fallback:
  `egl`); both are software paths on this box.

## Task 1: Scaffold, dependencies, arena model XML

**Files:** `.gitignore`, `requirements.txt`, `assets/dodge_humanoid.xml`,
`tests/test_model.py`.

1. `.gitignore`: `.venv/`, `runs/`, `__pycache__/`, `*.pyc`, `.superpowers/`,
   `*.mp4`.
2. Create `/root/dodge-rl/.venv` with `python3 -m venv --system-site-packages
   .venv`; `nice -n 10 .venv/bin/pip install "gymnasium[mujoco]"
   stable-baselines3 pytest imageio imageio-ffmpeg`; then
   `.venv/bin/pip freeze --local > requirements.txt` (`--local` is required: a
   plain freeze in a `--system-site-packages` venv captures the whole system
   python, including a `file:///tmp` torch wheel pin that would violate the
   torch constraint). Verify `import torch` in the venv reports 2.13.0+cpu
   (from system site-packages).
3. `assets/dodge_humanoid.xml`: copy the installed Gymnasium humanoid model
   (`.venv/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/humanoid.xml`)
   and modify:
   - In the `<default>` block, set the default geom `conaffinity="3"` so
     humanoid body geoms accept contacts from both the floor (contype 1) and
     projectiles (contype 2). Give the floor geom explicit `contype="1"
     conaffinity="1"`.
   - Add 4 projectile bodies at the worldbody level, named `proj0..proj3`:
     each a `<body>` with a `<freejoint>`, `gravcomp="1"` (straight flight —
     gravity fully compensated), one sphere geom `size="0.08"`, `mass="0.15"`,
     `contype="2"` `conaffinity="0"` (collides with humanoid geoms only — not
     floor, not other projectiles), distinct rgba per projectile, initial
     position `(0, 0, -10 - i)` (parked below the floor, inert).
   - Do NOT touch any joint `range`, actuator, or the humanoid body tree
     (constraint: anatomy unchanged).
4. `tests/test_model.py` (pytest, no Gymnasium env yet — raw `mujoco`):
   - model loads; `model.nbody` includes the 4 projectile bodies; each
     projectile has a free joint and `gravcomp == 1`.
   - joint ranges unchanged: assert `right_knee` range is (-160, -2) degrees
     (radians in-model: -2.792527, -0.034907, tolerance 1e-4) and `left_elbow`
     range is (-90, 50) degrees equivalent — read both from the compiled model.
   - contact filtering: place `proj0` overlapping the floor plane at
     `(2.5, 2.5, 0.05)` (outside the humanoid), `mj_forward` + `mj_step` a few
     steps, assert no contact involving `proj0` exists AND `proj0` does not
     decelerate (floor pass-through). Then place `proj0` overlapping the torso
     geom, step once, assert a contact pair (proj0 geom, humanoid geom) exists.
   - a projectile given velocity v keeps |velocity - v| < 1e-6 over 50 steps
     with no contact (gravcomp straight flight).

**Acceptance:** all tests green via `nice -n 10 .venv/bin/python -m pytest
tests/test_model.py -q`.

## Task 2: DodgeEnv — environment, spawner, reward, termination

**Files:** `dodge_rl/__init__.py`, `dodge_rl/dodge_env.py`,
`tests/test_env.py`.

`DodgeEnv(gymnasium.Env)` over `assets/dodge_humanoid.xml`, `frame_skip=5`
(control dt 0.015 s, physics dt 0.003 s), episode limit 2000 control steps
(30 s), Box action space = the 17 humanoid actuators with their stock
ctrlrange.

**Projectile slots.** 4 slots, matching `proj0..proj3`. Inactive slot: body
parked at its initial below-floor position with zero velocity (set qpos/qvel
directly). Active slot state tracked in the env (spawn time, TTL).

**Spawner** (all draws from `self.np_random`, per spawn event):
- Schedule: first spawn at t=0.5 s; subsequent inter-spawn intervals
  ~ U(1.0, 2.5) s; a spawn is skipped (rescheduled next step) while all 4
  slots are active.
- Spawn point: radius r ~ U(8, 15) m from current torso xyz; azimuth
  ~ U(0, 2π); elevation ~ U(-10°, +75°); if the resulting z would fall below
  0.3 m the point is raised to z = 0.3 and pushed outward along its azimuth so
  the torso distance stays exactly the drawn r (a literal z-clamp can shrink
  the distance below 8 m, violating the spawn-distance invariant).
- Target link: sampled from
  `[head, torso, pelvis(=root body), left_thigh, right_thigh, left_shin,
  right_shin]` with weights `[0.15, 0.30, 0.20, 0.10, 0.10, 0.075, 0.075]`;
  aim point = that body's current xpos.
- Precision: with p=0.5 aim exactly at the link centre; else add offset
  ~ N(0, 0.35²) m per axis.
- Speed ~ U(4, 9) m/s. Launch velocity = normalize(aim_point − spawn_point) ×
  speed, set once; never modified afterwards (no homing).
- TTL: despawn at age > 1.5 × (initial distance / speed), or when the
  projectile is > 20 m from the torso, or on hit.

**Observation** (`float64` Box): humanoid `qpos[2:]` (root x,y excluded) +
full humanoid `qvel` + per-slot block `[active_flag, rel_pos(3), rel_vel(3)]`
× 4 slots (rel_pos = projectile position − torso xpos; rel_vel = projectile
velocity − root translational velocity, i.e. both torso-relative; inactive
slots all-zero), slots ordered by
time-to-closest-approach ascending, inactive slots last. Projectile qpos/qvel
are excluded from the humanoid section (mask by joint address, don't assume
ordering).

**Reward per control step:**
- upright: `+1.0` when torso z ∈ [1.0, 2.0], else 0. (No fall termination.)
- proximity shaping: `−0.5 × Σ_active max(0, 1.2 − d_i)²` where `d_i` =
  distance from projectile i to the nearest of {head, torso, pelvis} xpos.
- hit: `−50` per contact between a projectile geom and any humanoid geom this
  step; the projectile despawns; episode CONTINUES.
- wall: `−100` and `terminated=True` when torso |x| > 1.5 or |y| > 1.5.
- control cost: `−0.05 × ||action||²`.
- info dict reports per-episode counters: `hits`, `wall_death` (bool),
  `spawns`, `min_approach` (closest any projectile ever got to the
  head/torso/pelvis set).

**Termination:** wall only. **Truncation:** 2000 steps.

**Registration:** `gymnasium.register(id="DodgeHumanoid-v0", ...)` in
`dodge_rl/__init__.py`, `max_episode_steps=2000`.

**Tests** (`tests/test_env.py`, TDD — written first, minimum):
- `gymnasium.utils.env_checker.check_env` passes.
- 100 spawns: all spawn distances in [8, 15] m from torso; all spawn z ≥ 0.3.
- Straight flight: an active projectile's velocity is constant (< 1e-6 drift)
  across ≥ 20 steps without contact.
- Precise-aim geometry: for a forced precise spawn, the line
  (spawn → launch velocity) passes within 1e-6 of the recorded aim point.
- Wall termination: teleport torso to x=1.6, step, assert terminated and
  reward ≤ −100 + step-level bounds.
- Hit: force a projectile into the torso, assert −50 applied, projectile
  despawned, episode not terminated.
- Determinism: two envs, same seed, same 200-step action sequence ⇒ identical
  observations and rewards.
- Slot ordering: construct two active projectiles with known
  closest-approach times, assert obs slot order.
- Anatomy guard: assert model joint ranges for `right_knee` match stock values
  (same numbers as Task 1's test — duplicated here so env-level refactors
  can't silently swap models).

**Acceptance:** full `tests/` suite green.

## Task 3: Training script — PPO on SubprocVecEnv

**Files:** `train.py`, `pyproject.toml`, `tests/test_train_smoke.py`.

- Minimal `pyproject.toml` (setuptools backend, package `dodge_rl`), then
  `nice -n 10 .venv/bin/pip install -e .` and regenerate `requirements.txt`
  via `pip freeze --local --exclude-editable`. This makes `dodge_rl`
  importable regardless of CWD — SubprocVecEnv worker processes and later
  eval tooling need it (Task 2 review demonstrated script-by-absolute-path
  fails without it).

- `train.py` (argparse): `--total-steps` (default 20_000_000), `--n-envs`
  (default 6), `--seed` (default 1), `--run-name` (default
  `ppo-{seed}-{total_steps}`), `--resume PATH` (checkpoint zip), `--out-dir`
  (default `runs/`).
- SB3 `PPO("MlpPolicy", ...)`: `policy_kwargs=dict(net_arch=[256, 256])`,
  `n_steps=2048`, `batch_size=4096`, `learning_rate=3e-4`, `gamma=0.99`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`, `device="cpu"`,
  `torch.set_num_threads(2)`.
- Env factory: `SubprocVecEnv` of `--n-envs` `DodgeHumanoid-v0`, wrapped in
  `VecMonitor` writing `monitor.csv` under the run dir; per-env seeds derived
  `seed + rank`.
- `CheckpointCallback` every 1_000_000 steps into the run dir; plus a custom
  callback printing every 50_000 steps: total steps, mean episode reward, mean
  episode length, mean hits/episode, wall-death rate (from `info` counters).
- On SIGTERM/SIGINT: save a final checkpoint `interrupt.zip` before exiting
  (SB3 model.save in a finally block is sufficient).
- `tests/test_train_smoke.py`: runs `train.py --total-steps 4096 --n-envs 2
  --run-name smoke` via subprocess (nice'd), asserts exit 0, a checkpoint zip
  and `monitor.csv` with ≥1 data row exist in `runs/smoke/`, then cleans
  `runs/smoke/`. Mark with `@pytest.mark.slow` and register the marker; it
  must still run (and pass) in the default suite.

**Acceptance:** suite green, including the smoke test.

## Task 4: Evaluation and headless video rendering

**Files:** `eval.py`, `render_video.py`, `tests/test_render.py`.

- `eval.py`: `--checkpoint PATH | --random`, `--episodes N` (default 20),
  `--seed`; runs deterministic policy episodes on a single env, prints a
  table: mean/std episode reward, mean episode length, hit rate (hits per
  episode), wall-death fraction, mean `min_approach`. Exits nonzero on
  no-checkpoint-found.
- `render_video.py`: `--checkpoint PATH | --random`, `--episodes` (default 2),
  `--out PATH.mp4`, `--fps 33`; offscreen-renders 640×480 frames
  (`mujoco.Renderer`, camera tracking the torso from outside the arena) and
  writes mp4 via imageio-ffmpeg. Must set `MUJOCO_GL=osmesa` (env var default,
  overridable); if osmesa init fails, retry with `egl` and report which
  backend worked. If a system package is missing (e.g. `libosmesa6`), install
  it via apt, and record the exact package list in the task report.
- `tests/test_render.py`: renders 30 frames from a random policy to a temp
  mp4; asserts the file exists, is > 10 kB, and `imageio` can read back ≥ 30
  frames. Marked slow like the train smoke test; still in the default suite.

**Acceptance:** suite green; a demo `demo-random.mp4` produced once by the
implementer (not committed — gitignored) and its path + backend named in the
task report.

## Out of plan scope

Launching the long training run, monitoring it, and iterating on reward
weights are lead/operator actions after the final review — not SDD tasks.
Reward-weight tuning from training results is expected follow-up work and not
a defect of the tasks above.
