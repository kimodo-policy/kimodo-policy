"""
Football physics material utilities.
UsdFileCfg does not support physics_material; apply it dynamically after resetting the scene.
The typical FIFA football restitution range is 0.7–0.8.
"""


def apply_football_physics_material(
    ball_prim_pattern: str = "/World/envs/env_.*/Object",
    restitution: float = 0.9,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.4,
) -> bool:
    """
    Apply a physics material to the football prim after scene loading to approximate realistic bounce.

    Run inside Isaac Sim / Isaac Lab with the isaaclab and omni modules available.

    Args:
        ball_prim_pattern: Football prim path or regex; default: /World/envs/env_.*/Object.
        restitution: Coefficient of restitution; typical FIFA range: 0.7–0.8.
        static_friction: Static friction.
        dynamic_friction: Dynamic friction.

    Returns:
        True on success; False if skipped or an error occurs.
    """
    try:
        import omni.usd
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
        from isaaclab.sim.utils import bind_physics_material

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[football_physics_material] ⚠️ No valid stage; skipping")
            return False

        matches = []
        try:
            from isaaclab.sim.utils.prims import find_matching_prim_paths

            found = find_matching_prim_paths(ball_prim_pattern)
            matches = list(found) if found else []
        except (ImportError, AttributeError):
            pass
        if not matches:
            for p in ["/World/envs/env_0/Object", "/World/envs/env_0/Football"]:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    matches = [p]
                    break
        if not matches:
            print(f"[football_physics_material] ⚠️ Football prim not found: {ball_prim_pattern}")
            return False

        material_path = "/World/FootballPhysicsMaterial"
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="max",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        spawn_rigid_body_material(material_path, mat_cfg)

        bound = 0
        for ball_path in matches:
            try:
                bind_physics_material(ball_path, material_path, stage=stage)
                bound += 1
            except Exception:
                pass

        if bound > 0:
            print(f"[football_physics_material] ✓ Applied restitution={restitution} to football ({bound} bindings)")
            return True
        print("[football_physics_material] ⚠️ Binding failed (football geometry may be instanced; edit the USD)")
        return False

    except ImportError as e:
        print(f"[football_physics_material] ⚠️ Required modules unavailable; skipping: {e}")
        return False
    except Exception as e:
        print(f"[football_physics_material] ⚠️ Failed to apply material: {e}")
        import traceback

        traceback.print_exc()
        return False