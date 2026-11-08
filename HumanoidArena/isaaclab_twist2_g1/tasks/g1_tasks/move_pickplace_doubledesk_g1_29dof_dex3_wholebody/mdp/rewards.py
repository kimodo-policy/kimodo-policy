# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Single-stage pick-place reward based only on goal placement.

Reward is **1** only while the hammer CoM is currently inside a goal bin region.
All previous lift-stage logic has been removed.

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
    BASKET_RIGID_SUBPRIM,
    CRATE_INIT_POS,
    HAMMER_INIT_POS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_DEFAULT_BASKET_CENTER_LOCAL = tuple(float(v) for v in CRATE_INIT_POS)
_DEFAULT_TABLE_SURFACE_Z = min(float(HAMMER_INIT_POS[2]), float(CRATE_INIT_POS[2]))

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
    """Approximate basket AABBs from task constants when USD inspection is unavailable."""
    pad_x = 0.05
    pad_y = 0.06
    pad_z_below = 0.04
    pad_z_above = 0.16
    cx, cy, z_base = _DEFAULT_BASKET_CENTER_LOCAL
    return [
        (
            cx - _CRATE_HALF_X_M - pad_x,
            cx + _CRATE_HALF_X_M + pad_x,
            cy - _CRATE_HALF_Y_M - pad_y,
            cy + _CRATE_HALF_Y_M + pad_y,
            z_base - pad_z_below,
            z_base + _CRATE_HEIGHT_M + pad_z_above,
        )
    ]


def _range_to_local_box(
    env: "ManagerBasedRLEnv",
    aligned_range,
) -> tuple[float, float, float, float, float, float] | None:
    origin0 = _scene_env_origin_w(env)[0].detach().cpu()
    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    sx = float(mx[0] - mn[0])
    sy = float(mx[1] - mn[1])
    sz = float(mx[2] - mn[2])
    if sx < 0.10 or sy < 0.10 or sz < 0.05 or sx > 2.0 or sy > 2.0 or sz > 1.0:
        return None
    return (
        float(mn[0] - origin0[0].item()),
        float(mx[0] - origin0[0].item()),
        float(mn[1] - origin0[1].item()),
        float(mx[1] - origin0[1].item()),
        float(mn[2] - origin0[2].item()),
        float(mx[2] - origin0[2].item()),
    )


def _load_basket_goal_boxes_from_stage_local(
    env: "ManagerBasedRLEnv",
    stage,
    bbox_cache,
) -> list[tuple[float, float, float, float, float, float]]:
    """Read the live imported ``/Basket`` AABB and convert it to env-local coordinates."""
    boxes: list[tuple[float, float, float, float, float, float]] = []
    candidate_paths = (
        "/World/envs/env_0/Basket",
        f"/World/envs/env_0/Basket/{BASKET_RIGID_SUBPRIM}",
    )
    basket_root = None
    for path in candidate_paths:
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid() or not prim.IsActive():
            continue
        if basket_root is None:
            basket_root = prim
        try:
            box = _range_to_local_box(env, bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is not None:
            boxes.append(box)

    if basket_root is None:
        return []

    if boxes:
        return _dedupe_goal_boxes_local(boxes)

    best_box = None
    best_volume = -1.0
    for prim in basket_root.GetChildren():
        if not prim.IsActive():
            continue
        try:
            box = _range_to_local_box(env, bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is None:
            continue
        volume = (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])
        if volume > best_volume:
            best_box = box
            best_volume = volume
    return [best_box] if best_box is not None else []


def _load_legacy_room_goal_boxes_from_stage_local(
    env: "ManagerBasedRLEnv",
    stage,
    bbox_cache,
) -> list[tuple[float, float, float, float, float, float]]:
    """Read legacy fixed-scene crate AABBs from the room USD."""
    env0_root = "/World/envs/env_0/Room"
    boxes: list[tuple[float, float, float, float, float, float]] = []
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
            box = _range_to_local_box(env, bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is not None:
            boxes.append(box)

    return _dedupe_goal_boxes_local(boxes)


def _get_goal_boxes_local(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float, float, float, float]]:
    if hasattr(env, "_ppdd_goal_boxes_local"):
        return env._ppdd_goal_boxes_local

    boxes: list[tuple[float, float, float, float, float, float]] = []
    source = "fallback"
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            boxes = _load_basket_goal_boxes_from_stage_local(env, stage, bbox_cache)
            if boxes:
                source = "basket_usd"
            else:
                boxes = _load_legacy_room_goal_boxes_from_stage_local(env, stage, bbox_cache)
                if boxes:
                    source = "legacy_room_usd"
    except Exception:
        boxes = []

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


def _ensure_rest_com_buf(env: "ManagerBasedRLEnv", n: int, device: torch.device) -> torch.Tensor:
    if not hasattr(env, "_ppdd_rest_com_w"):
        env._ppdd_rest_com_w = torch.zeros((n, 3), device=device, dtype=torch.float32)
    if not hasattr(env, "_ppdd_rest_com_ready"):
        env._ppdd_rest_com_ready = torch.zeros(n, dtype=torch.bool, device=device)
    return env._ppdd_rest_com_w


def _reset_stage_at_episode_start(
    env: "ManagerBasedRLEnv",
    obj: RigidObject,
) -> None:
    """Snapshot CoM on the desk at episode start (for debug baseline)."""
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
    ground_drop_margin_m: float = 0.30,
    goal_radius_xy: float = 0.0,
    target_z_margin_below_m: float = 0.0,
    target_z_margin_above_m: float = 0.0,
    reward_place: float = 1.0,
) -> torch.Tensor:
    """Stage-level reward using **CoM** (``root_com_pos_w``).

    Returns ``reward_place`` (1) only while CoM is currently inside a goal bin region.
    If the hammer falls to the ground, the reward is forced back to ``0``.

    :class:`~isaaclab.managers.RewardManager` multiplies by ``dt`` and term weight internally.
    Debug fields keep the raw additive values: goal=1, total=1.
    """

    obj = _get_hammer(env, object_cfg)
    _ensure_rest_com_buf(env, env.num_envs, env.device)
    _reset_stage_at_episode_start(env, obj)
    rest = _ensure_rest_com_buf(env, env.num_envs, env.device)
    _ensure_rest_baseline(env, obj, rest)

    com = obj.data.root_com_pos_w
    obj_x, obj_y, obj_z = com[:, 0], com[:, 1], com[:, 2]

    dz = obj_z - rest[:, 2]
    dxy = torch.norm(com[:, :2] - rest[:, :2], dim=1)
    table_surface_z = _DEFAULT_TABLE_SURFACE_Z
    on_ground = obj_z < (table_surface_z - ground_drop_margin_m)

    goal_boxes_local = _get_goal_boxes_local(env)

    in_goal = _in_any_goal_region(
        env,
        obj_x,
        obj_y,
        obj_z,
        goal_boxes_local=goal_boxes_local,
    )

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[in_goal] = reward_place
    reward[on_ground] = 0.0

    # One env: last metrics for get_reward_debug_string (after state update).
    if env.num_envs >= 1:
        i = 0
        step_dt = float(getattr(env, "step_dt", None) or (env.physics_dt * getattr(env.cfg, "decimation", 1)))
        dist_xy = _nearest_bin_dist_xy(env, obj_x, obj_y, goal_boxes_local)
        goal_debug_radius = _goal_debug_radius_xy(goal_boxes_local)
        goal_reward_value = float(reward_place if bool(in_goal[i].item()) else 0.0)
        raw_total = goal_reward_value
        if bool(on_ground[i].item()):
            goal_reward_value = 0.0
            raw_total = 0.0
        env._ppdd_dbg = {
            "com": (float(com[i, 0].item()), float(com[i, 1].item()), float(com[i, 2].item())),
            "rest": (float(rest[i, 0].item()), float(rest[i, 1].item()), float(rest[i, 2].item())),
            "dz": float(dz[i].item()),
            "dxy": float(dxy[i].item()),
            "on_ground": bool(on_ground[i].item()),
            "placed": bool(in_goal[i].item() and not on_ground[i].item()),
            "in_goal": bool(in_goal[i].item()),
            "dist_xy_bin": float(dist_xy[i].item()),
            "goal_radius_xy": float(goal_debug_radius),
            "goal_source": getattr(env, "_ppdd_goal_boxes_source", "unknown"),
            "goal_box_count": len(goal_boxes_local),
            "step_dt": step_dt,
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
        "_ppdd_prev_ep_len_buf",
        "_ppdd_goal_boxes_local",
        "_ppdd_goal_boxes_source",
        "_ppdd_dbg",
    ):
        if hasattr(env, name):
            delattr(env, name)


__all__ = ["compute_reward", "reset_ppdd_reward_cache"]
