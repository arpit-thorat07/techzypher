from pathlib import Path
import numpy as np
import trimesh


def validate_and_export(input_path, output_path):
    scene = trimesh.load(input_path, force="scene")

    if not scene.geometry:
        raise ValueError("Empty 3D model")

    for mesh in scene.geometry.values():
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Invalid mesh")

        if mesh.is_watertight and mesh.volume < 0:
            mesh.invert()

        if mesh.visual.kind == "vertex":
            colors = mesh.visual.vertex_colors.copy()
            material = trimesh.visual.material.PBRMaterial(
                name="Vertex Color Material",
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.8,
                doubleSided=True
            )
            visual = trimesh.visual.TextureVisuals(material=material)
            visual.vertex_attributes["color"] = colors
            mesh.visual = visual

    if scene.bounds is None or not np.isfinite(scene.bounds).all():
        raise ValueError("Invalid mesh bounds")

    if not np.all(scene.extents > 0):
        raise ValueError("Invalid 3D dimensions")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene.export(str(output_path))
    return str(output_path)
