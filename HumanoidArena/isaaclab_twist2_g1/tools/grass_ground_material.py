"""
Grass PBR ground material utilities.
Apply grass PBR textures to the ground prim after scene loading; run inside Isaac Sim.
Supports AITextured PBR texture naming (*__Png_albedo.png, *__Png_normal.png, etc.).
"""

import glob
import os
from pathlib import Path


def _find_texture(textures_dir: str, patterns: list[str]) -> str | None:
    """Search a directory using multiple patterns and return the first matching full path."""
    for pat in patterns:
        matches = glob.glob(os.path.join(textures_dir, pat))
        if matches:
            return matches[0]
    return None


def apply_grass_pbr_to_ground(
    prim_path: str = "/World/GroundPlane",
    textures_dir: str = None,
    uv_scale: tuple = (25.0, 25.0),
) -> bool:
    """
    Apply a grass PBR material to the specified ground prim.

    Run inside Isaac Sim with the pxr and omni modules available.
    Skip missing textures without interrupting scene execution.

    Args:
        prim_path: Ground prim path; default: /World/GroundPlane.
        textures_dir: Texture directory; defaults to PROJECT_ROOT/assets/objects/materials/grass_turf.
        uv_scale: UV repetition factors (scale_u, scale_v); larger values produce finer textures. The default (25, 25) scales grass appropriately for the robot.

    Returns:
        True if the PBR material is applied; False if textures are missing or an error occurs.
    """
    try:
        import omni.usd
        from pxr import Sdf, UsdShade, UsdGeom, Usd, Vt, Gf
    except ImportError as e:
        print(f"[grass_ground_material] ⚠️ Cannot import omni.usd or pxr; skipping grass material: {e}")
        return False

    # Resolve the texture directory
    if textures_dir is None:
        project_root = os.environ.get("PROJECT_ROOT")
        if not project_root:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        textures_dir = str(Path(project_root) / "assets" / "objects" / "materials" / "grass_turf")

    # Support AITextured naming (*__Png_albedo.png) and the legacy grass_albedo.png
    albedo_path = _find_texture(
        textures_dir,
        ["*__Png_albedo.png", "*_albedo.png", "grass_albedo.png"],
    )
    if not albedo_path:
        print(f"[grass_ground_material] ⚠️ Albedo texture not found; see assets/objects/materials/grass_turf/README.md")
        return False

    # Use an absolute path because Omniverse requires it to resolve textures
    albedo_path = os.path.abspath(albedo_path)
    textures_dir = os.path.abspath(textures_dir)
    print(f"[grass_ground_material] Texture directory: {textures_dir}")
    print(f"[grass_ground_material] Albedo: {albedo_path}")

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[grass_ground_material] ⚠️ Ground prim not found: {prim_path}")
        return False

    # Collect all meshes that need the material, including the prim and its descendants
    def _collect_geom_prims(p):
        prims = []
        for sub in Usd.PrimRange(p):
            if sub.IsA(UsdGeom.Mesh) or sub.IsA(UsdGeom.Cube) or sub.IsA(UsdGeom.Plane):
                prims.append(sub)
        return prims

    target_meshes = _collect_geom_prims(prim)
    if not target_meshes:
        target_meshes = [prim]

    def _ensure_st(mesh_prim):
        primvars = UsdGeom.PrimvarsAPI(mesh_prim)
        if mesh_prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                min_x = min(float(p[0]) for p in points)
                max_x = max(float(p[0]) for p in points)
                min_y = min(float(p[1]) for p in points)
                max_y = max(float(p[1]) for p in points)
                dx = max(max_x - min_x, 1e-6)
                dy = max(max_y - min_y, 1e-6)
                st_values = Vt.Vec2fArray(
                    [Gf.Vec2f((float(p[0]) - min_x) / dx, (float(p[1]) - min_y) / dy) for p in points]
                )
                st_primvar = primvars.GetPrimvar("st")
                if not st_primvar or not st_primvar.IsDefined():
                    st_primvar = primvars.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                    )
                st_primvar.Set(st_values)
                st_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
                return True
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            total = int(sum(face_counts)) if face_counts else 0
            if total <= 0:
                return False
            st_values = Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0)] * total)
            for i in range(total):
                u = 1.0 if (i % 4) in (1, 2) else 0.0
                v = 1.0 if (i % 4) in (2, 3) else 0.0
                st_values[i] = Gf.Vec2f(u, v)
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        if mesh_prim.IsA(UsdGeom.Cube):
            st_values = Vt.Vec2fArray(
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)] * 6
            )
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        if mesh_prim.IsA(UsdGeom.Plane):
            st_values = Vt.Vec2fArray(
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)]
            )
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        return False

    # Roughness texture (optional, for AITextured Roughness contrast 0.8).
    _roughness_path = _find_texture(
        textures_dir,
        ["*__Png_roughness.png", "*_roughness.png", "grass_roughness.png"],
    )
    roughness_path = os.path.abspath(_roughness_path) if _roughness_path else None

    # 1. Create a PBR material (UsdPreviewSurface + UsdUVTexture).
    mtl_path = Sdf.Path("/World/Looks/GrassTurfMaterial")
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    # Metallic 0.0 (grass is a dielectric).
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Albedo texture (sRGB color space)
    albedo_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("AlbedoTx"))
    albedo_tx.CreateIdAttr("UsdUVTexture")
    albedo_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(albedo_path)
    albedo_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    albedo_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo_out = albedo_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(albedo_out)

    # UV Scale: PrimvarReader -> UsdTransform2d (scale) -> texture coordinates st.
    st_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("StReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    uv_transform = UsdShade.Shader.Define(stage, mtl_path.AppendPath("UVTransform"))
    uv_transform.CreateIdAttr("UsdTransform2d")
    uv_transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader_out)
    uv_transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        (float(uv_scale[0]), float(uv_scale[1]))
    )
    uv_out = uv_transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    albedo_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)

    # Roughness texture (optional, AITextured Roughness contrast 0.8).
    if roughness_path:
        roughness_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("RoughnessTx"))
        roughness_tx.CreateIdAttr("UsdUVTexture")
        roughness_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(roughness_path)
        roughness_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # Roughness is grayscale and linear
        roughness_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
        roughness_out = roughness_tx.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(roughness_out)
    else:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)

    # Normal texture (optional, AITextured Normal strength 2.5).
    _normal_path = _find_texture(
        textures_dir,
        ["*__Png_normal.png", "*_normal.png", "grass_normal.png"],
    )
    normal_path = os.path.abspath(_normal_path) if _normal_path else None
    if normal_path:
        normal_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("NormalTx"))
        normal_tx.CreateIdAttr("UsdUVTexture")
        normal_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_path)
        normal_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # Normal is linear
        normal_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
        normal_out = normal_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Float3).ConnectToSource(normal_out)

    mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # 2. Bind the material to all ground meshes, replacing the default checker
    st_bound_count = 0
    for mesh_prim in target_meshes:
        if _ensure_st(mesh_prim):
            st_bound_count += 1

    bound_count = 0
    for mesh_prim in target_meshes:
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(mtl)
        bound_count += 1

    # 3. If prim_path is GroundPlane, also apply the material to defaultGroundPlane, which Isaac Sim may use
    if prim_path == "/World/GroundPlane":
        default_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if default_prim and default_prim.IsValid():
            for mesh_prim in _collect_geom_prims(default_prim):
                if _ensure_st(mesh_prim):
                    st_bound_count += 1
                binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                binding_api.Bind(mtl)
                bound_count += 1

    print(
        f"[grass_ground_material] ✅ Applied grass PBR material to {bound_count} meshes; "
        f"meshes with st UV coordinates: {st_bound_count}"
    )
    return True
