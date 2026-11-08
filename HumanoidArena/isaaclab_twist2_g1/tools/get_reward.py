#
# Utility example: how to get the reward
##
from __future__ import annotations

import torch


def _default_reward_tensor(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _env_reward_dt(env) -> float:
    """Environment control interval (matches ``ManagerBasedRLEnv`` reward ``dt``)."""
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    if hasattr(env, "physics_dt") and hasattr(env, "cfg"):
        return float(env.physics_dt) * int(getattr(env.cfg, "decimation", 1))
    return 0.02


def sync_task_events_after_physics_step(env) -> None:
    """Run task-local post-physics hooks when control code advances sim without ``env.step()``."""
    env_cfg = getattr(env, "cfg", None)
    if env_cfg is None:
        return
    apply_open_door_latch = getattr(env_cfg, "apply_open_door_latch_interval", None)
    if callable(apply_open_door_latch):
        try:
            apply_open_door_latch(env, reason="manual_post_physics")
        except Exception as exc:
            print(f"[task_events] open door latch sync failed: {exc}")
    debug_joint_runtime_step = getattr(env_cfg, "debug_joint_runtime_step", None)
    if callable(debug_joint_runtime_step):
        try:
            debug_joint_runtime_step(env)
        except Exception as exc:
            print(f"[task_events] open door joint debug sync failed: {exc}")


def sync_reward_after_physics_step(env) -> None:
    """Run once per control step after physics when ``env.step()`` is skipped (replay / wholebody).

    SONIC advances ``env.sim`` inside ``action_provider``; without this, ``reward_manager.compute``
    never runs, so sparse pick-place rewards never latch. Must use the same ``dt`` as
    ``ManagerBasedRLEnv.step()`` (``step_dt`` / ``physics_dt * decimation``).
    """
    sync_task_events_after_physics_step(env)
    rm = getattr(env, "reward_manager", None)
    if rm is None:
        return
    dt = _env_reward_dt(env)
    total = rm.compute(dt=dt)
    rb = getattr(env, "reward_buf", None)
    if rb is None or not isinstance(total, torch.Tensor):
        return
    try:
        if total.dtype != rb.dtype or total.device != rb.device:
            total = total.to(device=rb.device, dtype=rb.dtype)
        if total.shape == rb.shape:
            rb.copy_(total)
        elif total.numel() == rb.numel():
            rb.reshape(-1).copy_(total.reshape(-1))
    except Exception:
        pass


def _extract_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _extract_scalar(value[0])
    try:
        return float(value)
    except Exception:
        return 0.0


def get_step_reward_value(env) -> torch.Tensor:
    """
    Get the current environment's total reward efficiently.

    Prefer ``reward_buf`` written by the latest ``env.step()`` to avoid another call to
    ``reward_manager.compute()``. Repeated computation executes stateful reward functions twice;
    combined with Isaac Lab logic such as ``episode_length_buf == 1``, this can incorrectly clear phase flags.

    Returns:
        torch.Tensor: Current total reward, or a zero tensor on failure.
    """
    try:
        rb = getattr(env, "reward_buf", None)
        if rb is not None:
            return rb
        if hasattr(env, "reward_manager"):
            return env.reward_manager.compute(dt=_env_reward_dt(env))
        return _default_reward_tensor(env)
    except Exception as e:
        print(f"Failed to retrieve reward: {e}")
        return _default_reward_tensor(env)


def get_current_rewards(env):
    """
    Get the current environment reward.

    Prefer ``reward_buf`` (as in ``get_step_reward_value``) to avoid repeated computation.
    """
    try:
        rb = getattr(env, "reward_buf", None)
        if rb is not None:
            return rb
        if hasattr(env, "reward_manager"):
            return env.reward_manager.compute(dt=_env_reward_dt(env))
    except Exception as e:
        print(f"Failed to retrieve reward: {e}")
    return _default_reward_tensor(env)


def get_reward_debug_string(env) -> str:
    """
    Generate a readable reward debug string.

    The ``total`` field prefers the **raw, unscaled total** recorded by the reward function (for example, 0/3),
    rather than Isaac Lab's ``reward_buf``, which has already been multiplied by ``dt``.
    If the task provides no raw value, fall back to ``reward_buf`` / ``compute``.
    """
    if not hasattr(env, "reward_manager"):
        return "[reward_debug] reward_manager unavailable"

    try:
        dbg = getattr(env, "_ppdd_dbg", None)
        if dbg is None:
            dbg = getattr(env, "_ppbox_dbg", None)
        term_parts = []
        get_terms = getattr(env.reward_manager, "get_active_iterable_terms", None)
        if callable(get_terms):
            for entry in get_terms(0):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                name = str(entry[0])
                value = _extract_scalar(entry[1])
                term_parts.append(f"{name}={value:.4f}")

        if isinstance(dbg, dict) and "raw_total" in dbg:
            total_value = _extract_scalar(dbg["raw_total"])
        elif term_parts:
            total_value = sum(_extract_scalar(part.split("=")[1]) for part in term_parts if "=" in part)
        else:
            rb = getattr(env, "reward_buf", None)
            if rb is not None:
                total_value = _extract_scalar(rb)
            else:
                total_reward = env.reward_manager.compute(dt=_env_reward_dt(env))
                total_value = _extract_scalar(total_reward)
        base = (
            f"[reward_debug] total={total_value:.4f} | " + ", ".join(term_parts)
            if term_parts
            else f"[reward_debug] total={total_value:.4f}"
        )
        if isinstance(dbg, dict):
            if dbg.get("task") == "pickplace_box":
                com = dbg.get("com")
                if isinstance(com, (list, tuple)) and len(com) >= 3:
                    com_s = f"({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})"
                else:
                    com_s = "n/a"
                raw_support_z = dbg.get("support_top_z")
                target_support_z = dbg.get("target_support_top_z")
                base_support_z = dbg.get("base_support_top_z")
                raw_support_z_s = float(raw_support_z) if isinstance(raw_support_z, (int, float)) else float("nan")
                target_support_z_s = (
                    float(target_support_z) if isinstance(target_support_z, (int, float)) else float("nan")
                )
                base_support_z_s = (
                    float(base_support_z) if isinstance(base_support_z, (int, float)) else float("nan")
                )
                base = (
                    f"[reward_debug] total={_extract_scalar(dbg.get('raw_total', total_value)):.4f}"
                    f" | reward={_extract_scalar(dbg.get('goal_reward', total_value)):.4f}"
                    f" placed={int(bool(dbg.get('placed')))}"
                    f" source={dbg.get('surface_source', 'unknown')}"
                    f" surfaces={int(dbg.get('surface_count', 0))}"
                    f" surface_idx={int(dbg.get('surface_index', -1))}"
                    f" inside_xy={int(bool(dbg.get('inside_xy')))}"
                    f" aligned_z={int(bool(dbg.get('aligned_z')))}"
                    f" dx_out={float(dbg.get('dx_outside', float('nan'))):.4f}"
                    f" dy_out={float(dbg.get('dy_outside', float('nan'))):.4f}"
                    f" z_gap={float(dbg.get('z_gap', float('nan'))):.4f}"
                    f" bottom_z={float(dbg.get('box_bottom_z', float('nan'))):.4f}"
                    f" raw_support_z={raw_support_z_s:.4f}"
                    f" target_z={target_support_z_s:.4f}"
                    f" base_z={base_support_z_s:.4f}"
                    f" x=[{float(dbg.get('x_lo', float('nan'))):.3f},{float(dbg.get('x_hi', float('nan'))):.3f}]"
                    f" y=[{float(dbg.get('y_lo', float('nan'))):.3f},{float(dbg.get('y_hi', float('nan'))):.3f}]"
                    f" com={com_s}"
                )
                candidate_surfaces = dbg.get("candidate_surfaces")
                if isinstance(candidate_surfaces, list) and candidate_surfaces:
                    candidate_parts = []
                    for entry in candidate_surfaces[:3]:
                        if not isinstance(entry, dict):
                            continue
                        candidate_parts.append(
                            "cand("
                            f"idx={int(entry.get('surface_index', -1))},"
                            f"in={int(bool(entry.get('inside_xy')))},"
                            f"az={int(bool(entry.get('aligned_z')))},"
                            f"zg={float(entry.get('z_gap', float('nan'))):.3f},"
                            f"sz={float(entry.get('support_top_z', float('nan'))):.3f},"
                            f"tz={float(entry.get('target_support_top_z', float('nan'))):.3f}"
                            ")"
                        )
                    if candidate_parts:
                        base += " | " + " ".join(candidate_parts)
                return base

            com = dbg.get("com")
            if isinstance(com, (list, tuple)) and len(com) >= 3:
                com_s = f"({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})"
            else:
                com_s = "n/a"
            dist = dbg.get("dist_xy_bin")
            rad = dbg.get("goal_radius_xy")
            goal_source = dbg.get("goal_source")
            goal_box_count = dbg.get("goal_box_count")
            dist_s = ""
            if isinstance(dist, (int, float)) and isinstance(rad, (int, float)):
                dist_s = f" dist_xy_bin={float(dist):.3f}/{float(rad):.3f}"
            goal_s = ""
            if isinstance(goal_source, str):
                goal_s = f" goal={goal_source}"
                if isinstance(goal_box_count, int):
                    goal_s += f"({goal_box_count})"
            raw_total = _extract_scalar(dbg.get("raw_total", total_value))
            raw_reward = _extract_scalar(dbg.get("goal_reward", raw_total))
            base = f"[reward_debug] total={raw_total:.4f} | reward={raw_reward:.4f}"
            base += (
                f" dz={dbg.get('dz', float('nan')):.4f} dxy={dbg.get('dxy', float('nan')):.4f}"
                f" on_ground={int(bool(dbg.get('on_ground')))}"
                f" in_goal={int(bool(dbg.get('in_goal')))} placed={int(bool(dbg.get('placed')))}"
                f"{dist_s}{goal_s} com={com_s}"
            )
        return base
    except Exception as e:
        return f"[reward_debug] failed: {e}"
