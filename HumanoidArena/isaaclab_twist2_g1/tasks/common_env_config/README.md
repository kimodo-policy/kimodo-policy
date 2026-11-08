# Common Env Config YAML

`sim_main.py` supports environment overrides through `--env_config_yaml` before calling `gym.make(...)`.

Minimal configuration:

```yaml
task_name: Isaac-Move-Football-Single-G129-Dex3-Wholebody

overrides:
  sim:
    dt: 0.005
  decimation: 4
```

Additional overrides can be applied by route or task:

```yaml
task_name: Isaac-Move-Football-Single-G129-Dex3-Wholebody
backend: sonic

overrides:
  sim:
    dt: 0.005
  decimation: 4

routes:
  sonic:
    decimation: 8

tasks:
  Isaac-Move-Football-Single-G129-Dex3-Wholebody:
    episode_length_s: 30.0
```

Merge order:
`overrides` -> `routes.<gmt_backend|action_source>` -> `tasks.<task_name>` -> `route_tasks.<route>.<task>`

Launcher conventions:
- `run_sonic.sh` / `run_twist2.sh` read the Isaac task ID from the YAML top-level `task_name`.
- Switch tasks by changing only `ENV_CONFIG_YAML`.

Derived fields synchronized automatically:
- `scene.contact_forces.update_period <- sim.dt`
- `sim.render_interval <- decimation`

Recommended practices:
- Keep shared defaults in `twist2_default.yaml` / `sonic_default.yaml`.
- Use separate task files, such as `football_single_twist2.yaml`, `football_single_sonic.yaml`, or `football_single_vla.yaml`.
- Include `task_name` at the top level of every task YAML.
- Point run scripts directly at the task YAML instead of maintaining `TASK_NAME` separately.

For environment objects requiring deterministic reset, recording, or replay restoration, configure the following directly in the task YAML:

```yaml
overrides:
  sim:
    dt: 0.001
  decimation: 10
  object_reset_seed_source: time
  deterministic_object_resets:
    - record_name: football
      scene_keys: [object, football]
      pose_range:
        x: [-0.05, 0.05]
        y: [0.0, 0.05]
      zero_velocity_on_reset: true
```

Recommended seed sources:
- `object_reset_seed_source: time`
  Suitable for human demonstration collection. Each episode gets new local pseudorandom object states without affecting the global RNG.
- `object_reset_seed_source: env_seed`
  Suitable for VLA/policy evaluation. Under the same `--seed`, cold starts and reset cycles produce identical object spawn sequences for model comparisons.

Scene randomization rules:
- Each task must have only one master scene seed: `episode_object_seed`.
- Derive `target_sign`, obstacle layout, and task-specific scene initialization from this seed.
- Do not maintain a second layout seed for a local subsystem.
- Replay must prioritize the recorded `episode_object_seed` when rebuilding the scene.

Detailed constraints:
- [docs/SCENE_RANDOMIZATION_SEED_RULES.md](../../docs/SCENE_RANDOMIZATION_SEED_RULES.md)
