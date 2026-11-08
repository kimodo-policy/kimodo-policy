# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Two-stage pick-place reward (raw additive values).

1) **Lift**: once CoM leaves rest, lift reward is **1** every step until place.
2) **Place**: while **lifted** and CoM is **currently** inside a bin region, goal reward is **3**.
   So raw total is **1 + 3 = 4** while in goal, and drops back to **1** after leaving the bin.

Isaac Lab still multiplies by ``dt`` and term weight for ``reward_buf`` internally, but
``_ppdd_dbg['raw_total']`` keeps the unscaled task reward for debug printing.

Uses ``RigidObject.data.root_com_pos_w`` (CoM), **not** ``root_pos_w`` (actor/link frame). For a hammer the
actor origin can sit on the handle while the CoM is elsewhere; using the link frame makes lift/place
checks look “stuck” even when the task visually succeeds.

**Important:** ``sim_main`` debug must not call ``reward_manager.compute()`` twice; see ``tools/get_reward.py``.

**Wholebody / SONIC:** ``RobotController`` skips ``env.step()`` when ``replay_mode`` or ``use_rl_action_mode``
is set; physics runs in the action provider. ``sync_reward_after_physics_step`` in ``tools/get_reward.py`` must
run after each ``get_action`` so ``reward_manager.compute`` matches simulation (see ``robot_control_system.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from common_env_objects import resolve_env_object_scene_key
from tasks.common_scene.base_scene_pickplace_doubledesk import (
    CONTAINER_R_POS_OFFSET,
    OBJECT_L_POS_OFFSET,
    PACKING_TABLE_L_POS,
    PACKING_TABLE_R_POS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _all_bin_centers_local() -> list[tuple[float, float, float]]:
    """Bin centers on both desks (container + object offsets each)."""
    centers: list[tuple[float, float, float]] = []
    for ax, ay, az in (PACKING_TABLE_L_POS, PACKING_TABLE_R_POS):
        centers.append(
            (
                ax + CONTAINER_R_POS_OFFSET[0],
                ay + CONTAINER_R_POS_OFFSET[1],
                az + CONTAINER_R_POS_OFFSET[2],
            )
        )
        centers.append(
            (
                ax + OBJECT_L_POS_OFFSET[0],
                ay + OBJECT_L_POS_OFFSET[1],
                az + OBJECT_L_POS_OFFSET[2],
            )
        )
    out: list[tuple[float, float, float]] = []
    for c in centers:
        if not any(
            abs(c[0] - o[0]) < 1e-4 and abs(c[1] - o[1]) < 1e-4 and abs(c[2] - o[2]) < 1e-4
            for o in out
        ):
            out.append(c)
    return out


_CANDIDATE_CENTERS_LOCAL = _all_bin_centers_local()

_CRATE_HALF_X_M = 0.3007293
_CRATE_HALF_Y_M = 0.2008865
_CRATE_HEIGHT_M = 0.17000664


def _scene_env_origin_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    scene = getattr(env, "scene", None)
    if scene is None:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    origins = getattr(scene, "env_origins", None)
    if origins is None:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    return origins.to(device=env.device, dtype=torch.float32)


def _dedupe_goal_boxes_local(
    boxes: list[tuple[float, float, float, float, float, float]]
) -> list[tuple[float, float, float, float, float, float]]:
    out: list[tuple[float, float, float, float, float, float]] = []
    for box in boxes:
        cx = 0.5 * (box[0] + box[1])
        cy = 0.5 * (box[2] + box[3])
        cz = 0.5 * (box[4] + box[5])
        if any(
            abs(cx - 0.5 * (o[0] + o[1])) < 1e-3
            and abs(cy - 0.5 * (o[2] + o[3])) < 1e-3
            and abs(cz - 0.5 * (o[4] + o[5])) < 1e-3
            for o in out
        ):
            continue
        out.append(box)
    return out


def _fallback_goal_boxes_local() -> list[tuple[float, float, float, float, float, float]]:
    """Approximate yellow-crate AABBs from scene constants when USD inspection is unavailable."""
    pad_x = 0.05
    pad_y = 0.06
    pad_z_below = 0.04
    pad_z_above = 0.16
    boxes: list[tuple[float, float, float, float, float, float]] = []
    for cx, cy, z_base in _CANDIDATE_CENTERS_LOCAL:
        boxes.append(
            (
                cx - _CRATE_HALF_X_M - pad_x,
                cx + _CRATE_HALF_X_M + pad_x,
                cy - _CRATE_HALF_Y_M - pad_y,
                cy + _CRATE_HALF_Y_M + pad_y,
                z_base - pad_z_below,
                z_base + _CRATE_HEIGHT_M + pad_z_above,
            )
        )
    return _dedupe_goal_boxes_local(boxes)


def _load_goal_boxes_from_stage_local(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float, float, float, float]]:
    """Read yellow-crate world AABBs from the live USD stage and convert them to env-local boxes."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return []

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    env0_root = "/World/envs/env_0/Room"
    origin0 = _scene_env_origin_w(env)[0].detach().cpu()
    boxes: list[tuple[float, float, float, float, float, float]] = []
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
    except Exception:
        return []

    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        path = str(prim.GetPath())
        lower = path.lower()
        if not lower.startswith(env0_root.lower()):
            continue
        name = prim.GetName().lower()
        if not any(token in lower or token in name for token in ("sm_crate", "crate", "yellow", "a07")):
            continue
        try:
            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
        except Exception:
            continue

        sx = float(mx[0] - mn[0])
        sy = float(mx[1] - mn[1])
        sz = float(mx[2] - mn[2])
        if sx < 0.10 or sy < 0.10 or sz < 0.05 or sx > 2.0 or sy > 2.0 or sz > 1.0:
            continue

        boxes.append(
            (
                float(mn[0] - origin0[0].item()),
                float(mx[0] - origin0[0].item()),
                float(mn[1] - origin0[1].item()),
                float(mx[1] - origin0[1].item()),
                float(mn[2] - origin0[2].item()),
                float(mx[2] - origin0[2].item()),
            )
        )

    return _dedupe_goal_boxes_local(boxes)


def _get_goal_boxes_local(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float, float, float, float]]:
    if hasattr(env, "_ppdd_goal_boxes_local"):
        return env._ppdd_goal_boxes_local

    boxes = _load_goal_boxes_from_stage_local(env)
    source = "usd" if boxes else "fallback"
    if not boxes:
        boxes = _fallback_goal_boxes_local()

    env._ppdd_goal_boxes_local = boxes
    env._ppdd_goal_boxes_source = source
    return boxes


def _goal_debug_radius_xy(
    goal_boxes_local: list[tuple[float, float, float, float, float, float]]
) -> float:
    if not goal_boxes_local:
        return 0.0
    radius = 0.0
    for x_lo, x_hi, y_lo, y_hi, _z_lo, _z_hi in goal_boxes_local:
        half_x = 0.5 * (x_hi - x_lo)
        half_y = 0.5 * (y_hi - y_lo)
        radius = max(radius, float((half_x * half_x + half_y * half_y) ** 0.5))
    return radius


def _get_hammer(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> RigidObject:
    """Resolve ``object_l`` (hammer) scene key; falls back to ``object_cfg.name``."""
    name = object_cfg.name
    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        sk = resolve_env_object_scene_key(env, cfg, "object_l")
        if sk is not None:
            name = sk
    return env.scene[name]


def _get_lifted_flag(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "_ppdd_two_stage_lifted"):
        n = env.num_envs
        dev = env.device
        env._ppdd_two_stage_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    return env._ppdd_two_stage_lifted


def _ensure_rest_com_buf(env: "ManagerBasedRLEnv", n: int, device: torch.device) -> torch.Tensor:
    if not hasattr(env, "_ppdd_rest_com_w"):
        env._ppdd_rest_com_w = torch.zeros((n, 3), device=device, dtype=torch.float32)
    if not hasattr(env, "_ppdd_rest_com_ready"):
        env._ppdd_rest_com_ready = torch.zeros(n, dtype=torch.bool, device=device)
    return env._ppdd_rest_com_w


def _reset_stage_at_episode_start(
    env: "ManagerBasedRLEnv",
    lifted: torch.Tensor,
    obj: RigidObject,
) -> None:
    """Clear lift latch and snapshot CoM on the desk at episode start (for lift baseline)."""
    ep = getattr(env, "episode_length_buf", None)
    if ep is None:
        return
    if not hasattr(env, "_ppdd_prev_ep_len_buf"):
        env._ppdd_prev_ep_len_buf = torch.zeros_like(ep)
    prev = env._ppdd_prev_ep_len_buf
    prev_aligned = torch.where(ep < prev, torch.zeros_like(prev), prev)
    episode_start = (ep == 1) & (prev_aligned == 0)
    # Second reward_manager.compute in the same control step is suppressed by _ppdd_prev_ep_len_buf
    # (prev becomes ep before the second call).
    if episode_start.any():
        lifted[episode_start] = False
        rest = _ensure_rest_com_buf(env, env.num_envs, env.device)
        com = obj.data.root_com_pos_w
        rest[episode_start] = com[episode_start].detach()
        env._ppdd_rest_com_ready[episode_start] = True
    env._ppdd_prev_ep_len_buf = ep.clone()


def _ensure_rest_baseline(env: "ManagerBasedRLEnv", obj: RigidObject, rest: torch.Tensor) -> None:
    """First time we see an env, lock rest CoM to current pose so dz/dxy are not vs zeros."""
    com = obj.data.root_com_pos_w
    need = ~env._ppdd_rest_com_ready
    if need.any():
        rest[need] = com[need].detach()
        env._ppdd_rest_com_ready[need] = True


def _in_any_goal_region(
    env: "ManagerBasedRLEnv",
    obj_x: torch.Tensor,
    obj_y: torch.Tensor,
    obj_z: torch.Tensor,
    *,
    goal_boxes_local: list[tuple[float, float, float, float, float, float]],
) -> torch.Tensor:
    """True if CoM is inside any goal AABB, with box coordinates defined in env-local frame."""
    o = _scene_env_origin_w(env)
    ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
    inside = torch.zeros_like(obj_x, dtype=torch.bool)
    for x_lo, x_hi, y_lo, y_hi, z_lo, z_hi in goal_boxes_local:
        ix = (obj_x >= (x_lo + ox)) & (obj_x <= (x_hi + ox))
        iy = (obj_y >= (y_lo + oy)) & (obj_y <= (y_hi + oy))
        iz = (obj_z >= (z_lo + oz)) & (obj_z <= (z_hi + oz))
        inside |= ix & iy & iz
    return inside


def _nearest_bin_dist_xy(
    env: "ManagerBasedRLEnv",
    obj_x: torch.Tensor,
    obj_y: torch.Tensor,
    goal_boxes_local: list[tuple[float, float, float, float, float, float]],
) -> torch.Tensor:
    """Min distance in XY from CoM to any goal-box center (world frame)."""
    o = _scene_env_origin_w(env)
    ox, oy = o[:, 0], o[:, 1]
    min_d = torch.full_like(obj_x, float("inf"))
    for x_lo, x_hi, y_lo, y_hi, _z_lo, _z_hi in goal_boxes_local:
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        cx_w = cx + ox
        cy_w = cy + oy
        dx = obj_x - cx_w
        dy = obj_y - cy_w
        d = torch.sqrt(dx * dx + dy * dy)
        min_d = torch.minimum(min_d, d)
    return min_d


def compute_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_l"),
    lift_above_rest_m: float = 0.03,
    lift_xy_move_m: float = 0.08,
    ground_drop_margin_m: float = 0.30,
    goal_radius_xy: float = 0.0,
    target_z_margin_below_m: float = 0.0,
    target_z_margin_above_m: float = 0.0,
    reward_lift: float = 1.0,
    reward_place: float = 3.0,
) -> torch.Tensor:
    """Stage-level reward using **CoM** (``root_com_pos_w``).

    Returns ``0`` before first lift; ``reward_lift`` (1) after lift while **not** in bin;
    ``reward_lift + reward_place`` (4) while **lifted and** CoM is **currently** in a bin region.

    :class:`~isaaclab.managers.RewardManager` multiplies by ``dt`` and term weight internally.
    Debug fields keep the raw additive values: lift=1, goal=3, total=4.
    """

    obj = _get_hammer(env, object_cfg)
    lifted = _get_lifted_flag(env)
    _ensure_rest_com_buf(env, env.num_envs, env.device)
    _reset_stage_at_episode_start(env, lifted, obj)
    rest = _ensure_rest_com_buf(env, env.num_envs, env.device)
    _ensure_rest_baseline(env, obj, rest)

    com = obj.data.root_com_pos_w
    obj_x, obj_y, obj_z = com[:, 0], com[:, 1], com[:, 2]

    dz = obj_z - rest[:, 2]
    dxy = torch.norm(com[:, :2] - rest[:, :2], dim=1)
    # Lift: CoM clearly above rest, or moved sideways off the pickup spot (grasp + carry).
    is_lifted = (dz > lift_above_rest_m) | (dxy > lift_xy_move_m)
    table_surface_z = min(c[2] for c in _CANDIDATE_CENTERS_LOCAL)
    on_ground = obj_z < (table_surface_z - ground_drop_margin_m)

    goal_boxes_local = _get_goal_boxes_local(env)

    in_goal = _in_any_goal_region(
        env,
        obj_x,
        obj_y,
        obj_z,
        goal_boxes_local=goal_boxes_local,
    )

    new_lift = (~lifted) & is_lifted
    lifted[new_lift] = True

    in_box = lifted & in_goal
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[in_box] = reward_lift + reward_place
    reward[lifted & ~in_box] = reward_lift
    reward[on_ground] = 0.0

    # One env: last metrics for get_reward_debug_string (after state update).
    if env.num_envs >= 1:
        i = 0
        step_dt = float(getattr(env, "step_dt", None) or (env.physics_dt * getattr(env.cfg, "decimation", 1)))
        dist_xy = _nearest_bin_dist_xy(env, obj_x, obj_y, goal_boxes_local)
        goal_debug_radius = _goal_debug_radius_xy(goal_boxes_local)
        lift_reward_value = float(reward_lift if bool(lifted[i].item()) else 0.0)
        goal_reward_value = float(reward_place if bool(in_box[i].item()) else 0.0)
        raw_total = lift_reward_value + goal_reward_value
        if bool(on_ground[i].item()):
            lift_reward_value = 0.0
            goal_reward_value = 0.0
            raw_total = 0.0
        env._ppdd_dbg = {
            "com": (float(com[i, 0].item()), float(com[i, 1].item()), float(com[i, 2].item())),
            "rest": (float(rest[i, 0].item()), float(rest[i, 1].item()), float(rest[i, 2].item())),
            "dz": float(dz[i].item()),
            "dxy": float(dxy[i].item()),
            "on_ground": bool(on_ground[i].item()),
            "is_lifted": bool(is_lifted[i].item()),
            "lifted": bool(lifted[i].item()),
            "placed": bool(in_box[i].item()),
            "in_goal": bool(in_goal[i].item()),
            "dist_xy_bin": float(dist_xy[i].item()),
            "goal_radius_xy": float(goal_debug_radius),
            "goal_source": getattr(env, "_ppdd_goal_boxes_source", "unknown"),
            "goal_box_count": len(goal_boxes_local),
            "step_dt": step_dt,
            "lift_reward": lift_reward_value,
            "goal_reward": goal_reward_value,
            "raw_total": raw_total,
        }

    return reward


def reset_ppdd_reward_cache(env: "ManagerBasedRLEnv") -> None:
    """Clear pick-place reward state after ``reset_object_self`` / ``reset_all_self``.

    The whole-body path often bypasses ``env.step()``, so ``episode_length_buf`` does not enter the ``ep==1`` branch.
    After an event resets an object, refresh its rest baseline and phase flags manually; otherwise, ``dz``/``dxy`` use the wrong reference pose.
    """
    for name in (
        "_ppdd_rest_com_w",
        "_ppdd_rest_com_ready",
        "_ppdd_two_stage_lifted",
        "_ppdd_prev_ep_len_buf",
        "_ppdd_goal_boxes_local",
        "_ppdd_goal_boxes_source",
        "_ppdd_dbg",
    ):
        if hasattr(env, name):
            delattr(env, name)


__all__ = ["compute_reward", "reset_ppdd_reward_cache"]
