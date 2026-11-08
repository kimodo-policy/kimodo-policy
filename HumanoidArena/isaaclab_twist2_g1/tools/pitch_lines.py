"""
Football pitch marking utilities.
Create white ground markings dynamically after scene loading; run inside Isaac Sim.

Coordinates (Isaac Sim, Z-up): ground in XY, height along Z.
Important: if the USD stage uses metersPerUnit=0.01 (centimeters), meters are converted to USD units automatically.
"""

import math
from typing import Tuple

# Line width and thickness (m)
DEFAULT_LINE_WIDTH = 0.12
DEFAULT_LINE_HEIGHT = 0.005
DEFAULT_LINE_COLOR = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0)

# Compact field layout synced with base_scene_football_cfg_wholebody (goal_y is the goal y; grass_half is half the grass width)
def _get_football_layout():
    """Read the layout from scene configuration, using fallback values on failure."""
    try:
        from tasks.common_scene.base_scene_football_cfg_wholebody import (
            GOAL_DISTANCE,
            ROBOT_INIT_Y,
        )
        goal_y = ROBOT_INIT_Y + GOAL_DISTANCE
        grass_half = 7.0  # ground size (14,14) / 2
        return goal_y, grass_half
    except ImportError:
        return 6.0, 7.0

_LAYOUT_GOAL_Y, _LAYOUT_GRASS_HALF = _get_football_layout()
# Units are meters: the goal line is slightly in front of the goal, and sidelines are inset from the grass edge
SMALL_PITCH_GOAL_Y = _LAYOUT_GOAL_Y - 0.5
SMALL_PITCH_GRASS_HALF = _LAYOUT_GRASS_HALF - 0.5

# FIFA standard field dimensions in meters for create_pitch_lines_in_stage
FIFA_LINE_WIDTH = 0.12
FIFA_PITCH_LENGTH = 105.0
FIFA_PITCH_WIDTH = 68.0
FIFA_CENTER_RADIUS = 9.15
FIFA_CORNER_ARC_RADIUS = 1.0
FIFA_PENALTY_LENGTH = 16.5
FIFA_PENALTY_WIDTH = 40.3
FIFA_GOAL_LENGTH = 5.5
FIFA_GOAL_WIDTH = 18.32
FIFA_PENALTY_SPOT_DIST = 11.0
FIFA_PENALTY_ARC_RADIUS = 9.15


def _get_stage_scale(stage) -> float:
    """
    Get the stage unit scale; all input parameters are in meters.
    For metersPerUnit=1, 1 m equals 1 USD unit; for metersPerUnit=0.01 (centimeters), 1 m equals 100 USD units.
    Returns the conversion multiplier from meters to USD units (= 1/metersPerUnit).
    """
    try:
        root = stage.GetRootLayer()
        mpu = root.GetMetadataByKey("metersPerUnit") if root else None
        if mpu is not None:
            mpu_f = float(mpu)
            if mpu_f > 0 and mpu_f != 1.0:
                scale = 1.0 / mpu_f
                print(f"[pitch_lines] Detected metersPerUnit={mpu}; multiplied meter values by {scale:.0f} to convert to USD units")
                return scale
        return 1.0
    except Exception:
        return 1.0


def _check_stage_axes(stage) -> None:
    """Check stage units and upAxis, warning about unexpected settings."""
    try:
        root = stage.GetRootLayer()
        up = root.GetMetadataByKey("upAxis") if root else None
        if up is not None and str(up) != "Z":
            print(f"[pitch_lines] ⚠️ upAxis={up} (not Z-up); markings may be misaligned")
    except Exception:
        pass


def create_robot_circle(
    stage,
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 2.0,
    parent_path: str = "/World/PitchLines",
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    n_segments: int = 128,
) -> bool:
    """
    Draw a white circle around the robot (default radius: 2 m, configurable).
    Use an annular mesh with vertices along the arc to avoid jagged cube seams and inconsistent X/Y scaling.

    Args:
        stage: USD stage（omni.usd.get_context().get_stage()）
        center: Circle center (x, y); defaults to (0, 0), matching the robot's initial position.
        radius: Circle radius in meters.
        parent_path: Parent prim path.
        line_width: Line width in meters.
        line_height: Line thickness; slightly above the ground to avoid z-fighting.
        line_color: Marking color.
        n_segments: Number of arc segments; more segments give a smoother circle (default: 128).

    Returns:
        True on success.
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ Cannot import pxr; skipping: {e}")
        return False

    scale = _get_stage_scale(stage)
    _check_stage_axes(stage)

    w = line_width * scale
    h = line_height * scale
    cx, cy = center[0] * scale, center[1] * scale
    radius = radius * scale

    # Ring mesh: define the arc with inner and outer vertices in (x, y, z), without scaling or rotation
    r_in = radius - w / 2
    r_out = radius + w / 2
    z = h / 2

    points = []
    for i in range(n_segments + 1):
        a = 2 * math.pi * i / n_segments
        points.append(Gf.Vec3f(cx + r_in * math.cos(a), cy + r_in * math.sin(a), z))
        points.append(Gf.Vec3f(cx + r_out * math.cos(a), cy + r_out * math.sin(a), z))

    face_vertex_counts = []
    face_vertex_indices = []
    for i in range(n_segments):
        i0 = 2 * i
        i1 = 2 * (i + 1)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([i0, i0 + 1, i1 + 1, i1])

    path = f"{parent_path}/circle_ring"
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    else:
        for child in list(parent.GetChildren()):
            stage.RemovePrim(child.GetPath())

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])

    mat_path = f"{path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(mat)

    print(
        f"[pitch_lines] Created annular mesh: center {center}, radius {radius}m, line width {w}m, "
        f"{n_segments} segments, line_color={line_color}"
    )
    return True


def _create_rect_ring_mesh(
    stage, parent_path: str, name: str,
    gx: float, gy: float, w: float, z: float,
    line_color: Tuple[float, float, float],
) -> None:
    """
    Create a rectangular ring mesh with uniform width and nonoverlapping corners as a single geometry.
    gx, gy = half-width and half-length; w = line width.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    # Four outer corners (counterclockwise)
    o0 = Gf.Vec3f(-gx - w / 2, -gy - w / 2, z)
    o1 = Gf.Vec3f(gx + w / 2, -gy - w / 2, z)
    o2 = Gf.Vec3f(gx + w / 2, gy + w / 2, z)
    o3 = Gf.Vec3f(-gx - w / 2, gy + w / 2, z)
    # Four inner corners (counterclockwise)
    i0 = Gf.Vec3f(-gx + w / 2, -gy + w / 2, z)
    i1 = Gf.Vec3f(gx - w / 2, -gy + w / 2, z)
    i2 = Gf.Vec3f(gx - w / 2, gy - w / 2, z)
    i3 = Gf.Vec3f(-gx + w / 2, gy - w / 2, z)
    points = Vt.Vec3fArray([o0, o1, o2, o3, i0, i1, i2, i3])
    face_vertex_counts = [4, 4, 4, 4]
    # Vertex order points normals toward +Z so the surface is visible from above (avoids backface culling)
    face_vertex_indices = [
        0, 4, 5, 1,   # Bottom edge
        1, 5, 6, 2,   # Right edge
        2, 6, 7, 3,   # Top edge
        3, 7, 4, 0,   # Left edge
    ]
    path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])
    mat_path = f"{path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(mat)


def create_small_pitch_boundary(
    stage,
    parent_path: str = "/World/PitchLines",
    goal_y: float = SMALL_PITCH_GOAL_Y,
    grass_half: float = SMALL_PITCH_GRASS_HALF,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    clear_parent: bool = False,
) -> bool:
    """
    Create four boundary lines for a small pitch. Parameters are in meters and converted using the stage's metersPerUnit.
    - Goal line 1: y=goal_y (goal side), along X.
    - Goal line 2: y=-goal_y, along X.
    - Touchline 1: x=-grass_half (grass edge), along Y.
    - Touchline 2: x=+grass_half, along Y.
    """
    try:
        from pxr import UsdGeom
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ Cannot import pxr: {e}")
        return False

    scale = _get_stage_scale(stage)
    w = line_width * scale
    h = line_height * scale
    goal_y_u = goal_y * scale
    grass_half_u = grass_half * scale
    z = h / 2

    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    elif clear_parent:
        for child in list(parent.GetChildren()):
            stage.RemovePrim(child.GetPath())

    # Single rectangular ring mesh: one connected piece with no overlapping corners
    _create_rect_ring_mesh(
        stage, parent_path, "pitch_boundary",
        grass_half_u, goal_y_u, w, z,
        line_color,
    )

    print(f"[pitch_lines] Created pitch boundaries: {2*grass_half:.1f}m × {2*goal_y:.1f}m (goal lines y=±{goal_y}, touchlines x=±{grass_half})")
    return True


def create_goal_reference_lines(
    stage,
    parent_path: str = "/World/PitchLines",
    goal_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2.5, 3.0), (2.5, -3.0)),
    goal_relative_offsets: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, -0.45), (0.0, 0.45)),
    goal_line_length: float = 4.0,
    goal_line_width: float = DEFAULT_LINE_WIDTH * 0.5,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bool:
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ Cannot import pxr: {e}")
        return False

    scale = _get_stage_scale(stage)
    line_len_u = goal_line_length * scale
    line_w_u = goal_line_width * scale
    line_h_u = line_height * scale
    z = line_h_u / 2

    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    for idx, ((goal_x, goal_y), (rel_x, rel_y)) in enumerate(zip(goal_centers, goal_relative_offsets), start=1):
        px = (goal_x + rel_x) * scale
        py = (goal_y + rel_y) * scale
        path = f"{parent_path}/goal_reference_line_{idx}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3d(line_len_u, line_w_u, line_h_u))
        trans_op = xform.AddTranslateOp()
        trans_op.Set(Gf.Vec3d(px, py, z))
        mat_path = f"{path}_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(mat)
        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_api.CreateCollisionEnabledAttr(False)

    print(
        f"[pitch_lines] Created goal reference line: length={goal_line_length}m, width={goal_line_width}m, "
        f"offsets={goal_relative_offsets}, centers={goal_centers}"
    )
    return True


def create_simple_debug_lines(
    stage,
    parent_path: str = "/World/PitchLines",
    center: Tuple[float, float] = (0.0, 0.0),
    circle_radius: float = 1.5,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    draw_pitch_boundary: bool = True,
    draw_goal_reference_lines: bool = False,
    goal_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2.5, 3.0), (2.5, -3.0)),
    goal_relative_offsets: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, -0.45), (0.0, 0.45)),
    goal_line_length: float = 2.0,
    goal_line_width: float = DEFAULT_LINE_WIDTH * 0.5,
    goal_line_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs,
) -> bool:
    """
    Create markings: a circle around the robot and four pitch boundaries.
    Keep this public function name for compatibility with existing callers (sim_main and test_* scripts).
    """
    ok1 = create_robot_circle(
        stage,
        center=center,
        radius=circle_radius,
        parent_path=parent_path,
        line_width=line_width,
        line_height=line_height,
        line_color=line_color,
    )
    ok2 = True
    ok3 = True
    if draw_goal_reference_lines:
        ok3 = create_goal_reference_lines(
            stage,
            parent_path=parent_path,
            goal_centers=goal_centers,
            goal_relative_offsets=goal_relative_offsets,
            goal_line_length=goal_line_length,
            goal_line_width=goal_line_width,
            line_height=line_height,
            line_color=goal_line_color,
        )
    return ok1 and ok2 and ok3


def create_pitch_lines_in_stage(
    stage,
    parent_path: str = "/World/PitchLines",
    center: Tuple[float, float] = (0.0, 0.0),
    scale: float = 0.17,
    line_width: float = FIFA_LINE_WIDTH,
    line_height: float = 0.005,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    axis_mode: str = "z_up",
) -> bool:
    """
    Create complete white FIFA-standard football pitch markings in a USD stage.
    scale reduces pitch dimensions for a smaller pitch; line_width is not scaled,
    so markings remain visible. Ball and goal dimensions remain unchanged; only the pitch layout is scaled.

    Run inside Isaac Sim with the pxr and omni modules available.

    Args:
        stage: USD stage, for example omni.usd.get_context().get_stage().
        parent_path: Parent prim path for all markings.
        center: Pitch center (x, y).
        scale: Scale factor for FIFA pitch dimensions; the goal and ball remain unchanged.
        line_width: Marking width in meters; default 0.12, independent of scale to preserve visibility.
        line_height: Marking thickness; slightly above the ground to avoid z-fighting.
        line_color: RGB marking color; white by default.
        axis_mode: "z_up" uses the XY ground plane (Isaac Sim default); "y_up" uses the XZ ground plane.

    Returns:
        True on success; False on error.
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ Cannot import pxr; skipping markings: {e}")
        return False

    w = line_width  # Do not scale with the field; keep the normal width
    h = line_height
    cx, cy = center

    # Half-length and half-width of the field
    half_len = (FIFA_PITCH_LENGTH / 2) * scale
    half_wid = (FIFA_PITCH_WIDTH / 2) * scale
    penalty_len = FIFA_PENALTY_LENGTH * scale
    penalty_wid = (FIFA_PENALTY_WIDTH / 2) * scale
    goal_len = FIFA_GOAL_LENGTH * scale
    goal_wid = (FIFA_GOAL_WIDTH / 2) * scale
    center_radius = FIFA_CENTER_RADIUS * scale
    corner_arc_radius = FIFA_CORNER_ARC_RADIUS * scale
    penalty_spot_dist = FIFA_PENALTY_SPOT_DIST * scale
    penalty_arc_radius = FIFA_PENALTY_ARC_RADIUS * scale

    def _add_arc_segments(name_prefix: str, arc_cx: float, arc_cy: float, radius: float,
                          angle_start: float, angle_end: float, n_seg: int = 12):
        """Draw an arc approximated by line segments, from angle_start to angle_end in radians."""
        for i in range(n_seg):
            t1 = i / n_seg
            t2 = (i + 1) / n_seg
            a1 = angle_start + (angle_end - angle_start) * t1
            a2 = angle_start + (angle_end - angle_start) * t2
            x1 = arc_cx + radius * math.cos(a1)
            y1 = arc_cy + radius * math.sin(a1)
            x2 = arc_cx + radius * math.cos(a2)
            y2 = arc_cy + radius * math.sin(a2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            seg_len = 2 * radius * math.sin((a2 - a1) / 2)
            seg_angle = (a1 + a2) / 2 + math.pi / 2
            _add_line(f"{name_prefix}_{i}", seg_len, w, mid_x, mid_y, seg_angle)

    def _add_line(name: str, length: float, width: float, px: float, py: float, angle: float):
        """Create a single marking cube. length follows the line, width is line width, and h is thickness. angle is rotation around the normal in radians.
        z_up: XY ground, Z normal; y_up: XZ ground, Y normal.
        """
        path = f"{parent_path}/{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)  # Unit cube from (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)
        xform = UsdGeom.Xformable(cube.GetPrim())
        if axis_mode == "y_up":
            # Ground lies in XZ with Y up. Lay lines on XZ, scale by (length_x, h, width_z), and rotate about Y
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(length, h, width))
            rot_op = xform.AddRotateYOp()
            rot_op.Set(math.degrees(angle))
            trans_op = xform.AddTranslateOp()
            trans_op.Set(Gf.Vec3d(px, h / 2, py))
        else:
            # Ground lies in XY with Z up (Isaac Sim default)
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(length, width, h))
            rot_op = xform.AddRotateZOp()
            rot_op.Set(math.degrees(angle))
            trans_op = xform.AddTranslateOp()
            trans_op.Set(Gf.Vec3d(px, py, h / 2))
        # Material
        mat_path = f"{path}_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(mat)
        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])

    # Ensure the parent prim exists under env_0 at the same level as the ball and goal for viewport rendering
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    # Sidelines and goal lines (four outer edges)
    # Goal lines (68 m): at Y = cy +/- half_len, along X
    _add_line("line_goal_top", FIFA_PITCH_WIDTH * scale, w, cx, cy + half_len, 0.0)
    _add_line("line_goal_bottom", FIFA_PITCH_WIDTH * scale, w, cx, cy - half_len, 0.0)
    # Sidelines (105 m): at X = cx +/- half_wid, along Y
    _add_line("line_touch_left", FIFA_PITCH_LENGTH * scale, w, cx - half_wid, cy, math.pi / 2)
    _add_line("line_touch_right", FIFA_PITCH_LENGTH * scale, w, cx + half_wid, cy, math.pi / 2)

    # Halfway line along X, 68 m long
    _add_line("line_center", FIFA_PITCH_WIDTH * scale, w, cx, cy, 0.0)

    # Center circle (approximated with line segments)
    n_arc = 24
    for i in range(n_arc):
        a1 = 2 * math.pi * i / n_arc
        a2 = 2 * math.pi * (i + 1) / n_arc
        x1 = cx + center_radius * math.cos(a1)
        y1 = cy + center_radius * math.sin(a1)
        x2 = cx + center_radius * math.cos(a2)
        y2 = cy + center_radius * math.sin(a2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        seg_len = 2 * center_radius * math.sin(math.pi / n_arc)
        seg_angle = (a1 + a2) / 2 + math.pi / 2
        _add_line(f"center_arc_{i}", seg_len, w, mid_x, mid_y, seg_angle)

    # Penalty and goal areas at both ends (penalty area: 16.5 m deep x 40.3 m wide; goal area: 5.5 m deep x 18.32 m wide)
    # top is the +Y end (attacking goal); bottom is the -Y end
    for side, sgn in [("bottom", -1), ("top", 1)]:
        goal_line_y = cy + sgn * half_len
        pen_front_y = goal_line_y - sgn * penalty_len  # Penalty-area fronts face the halfway line
        pen_center_y = (goal_line_y + pen_front_y) / 2
        # Penalty-area front (parallel to the goal line, along X) and side lines (along Y)
        _add_line(f"penalty_front_{side}", penalty_wid * 2, w, cx, pen_front_y, 0.0)
        _add_line(f"penalty_side_left_{side}", penalty_len, w, cx - penalty_wid, pen_center_y, math.pi / 2)
        _add_line(f"penalty_side_right_{side}", penalty_len, w, cx + penalty_wid, pen_center_y, math.pi / 2)
        # Goal area
        goal_front_y = goal_line_y - sgn * goal_len
        goal_center_y = (goal_line_y + goal_front_y) / 2
        _add_line(f"goal_front_{side}", goal_wid * 2, w, cx, goal_front_y, 0.0)
        _add_line(f"goal_side_left_{side}", goal_len, w, cx - goal_wid, goal_center_y, math.pi / 2)
        _add_line(f"goal_side_right_{side}", goal_len, w, cx + goal_wid, goal_center_y, math.pi / 2)

        # Penalty arc (9.15 m radius around the penalty spot; draw only the arc outside the penalty area, facing midfield)
        penalty_spot_y = goal_line_y - sgn * penalty_spot_dist
        d_over_r = (penalty_len - penalty_spot_dist) / penalty_arc_radius  # 5.5/9.15
        d_over_r = min(max(d_over_r, 0.0), 1.0)
        alpha = math.asin(d_over_r)
        if sgn > 0:  # top goal: arc faces midfield (-Y), so sin(theta) < -d/r
            arc_start = math.pi + alpha
            arc_end = 2 * math.pi - alpha
        else:  # bottom goal: arc faces midfield (+Y), so sin(theta) > d/r
            arc_start = alpha
            arc_end = math.pi - alpha
        _add_arc_segments(f"penalty_arc_{side}", cx, penalty_spot_y, penalty_arc_radius, arc_start, arc_end, n_seg=16)

    # Corner arcs (one quarter-circle at each corner, radius 1 m, inside the field)
    # Corner coordinates: (x, y) = (cx +/- half_wid, cy +/- half_len)
    corners = [
        ("tl", cx - half_wid, cy + half_len, 0, -math.pi / 2),           # Top left
        ("tr", cx + half_wid, cy + half_len, math.pi, math.pi / 2),       # Top right
        ("bl", cx - half_wid, cy - half_len, -math.pi / 2, 0),            # Bottom left
        ("br", cx + half_wid, cy - half_len, math.pi / 2, math.pi),       # Bottom right
    ]
    for cname, corn_x, corn_y, a_start, a_end in corners:
        _add_arc_segments(f"corner_arc_{cname}", corn_x, corn_y, corner_arc_radius, a_start, a_end, n_seg=8)

    print(f"[pitch_lines] Created full FIFA markings at {parent_path}, center {center}, scale {scale} "
          f"(pitch approximately {FIFA_PITCH_LENGTH*scale:.1f}m x {FIFA_PITCH_WIDTH*scale:.1f}m, line width {w*100:.1f}cm)")
    return True
