# Scene Randomization Seed Rules

These rules apply to all tasks with randomized scene initialization in `isaaclab_twist2_g1`, preventing recording and replay from following different randomization paths.

## Core Principle

Scene randomization must have a single master seed.

- Generate only one `episode_object_seed` per episode.
- Derive all environment object randomization from this seed.
- Record this master seed without depending on hidden task-local seeds.
- Replay must prioritize the recorded master seed when rebuilding the scene.

## Allowed Seed Sources

`episode_object_seed` may come from either of two sources:

- `object_reset_seed_source: time`
  Suitable for human demonstration collection. Each episode gets a new randomized scene.
- `object_reset_seed_source: env_seed`
  Suitable for evaluation and reproducible experiments. With the same `--seed`, both cold starts and reset sequences are reproducible.

These are two sources for the same master seed, not independent randomization paths.

## Disallowed Patterns

Do not maintain a second layout or scene seed for an individual subsystem, for example:

- A separate seed for obstacle layout
- A separate seed for the target object
- A newly invented task-local seed during replay restoration

Additional task-specific layout steps must use the same `episode_object_seed`.

## Task Implementation Requirements

Tasks with randomized scene initialization must:

1. Generate or assign the current `episode_object_seed` at live reset.
2. Derive all randomized task objects from that seed.
3. Save `episode_object_seed` and `episode_init_env` together in the recording NPZ.
4. Prioritize that seed when restoring task-specific scene initialization during replay.
5. Repeat any additional physics synchronization steps during replay restoration.

## Lessons from SmallWarehouse

`SmallWarehouse VisionNavigation` previously had the following issue:

- `target_sign` used the common deterministic reset seed chain.
- Obstacle layout used a separate task-local seed chain.
- Replay knew only `episode_object_seed`, not the obstacle layout seed.

As a result:

- Scene initialization differed between recording and replay.
- Even with object states in the NPZ, replay could restore an incorrect physical obstacle layout.

All new tasks must therefore follow the single-master-seed rule.
