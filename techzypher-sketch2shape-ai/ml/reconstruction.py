import os
from pathlib import Path
import subprocess

DEFAULT_TRIPOSR_ROOT = os.environ.get("TRIPOSR_ROOT", "/content/TripoSR")
DEFAULT_TRIPOSR_PYTHON = os.environ.get("TRIPOSR_PYTHON", "/content/triposr-env/bin/python")


def reconstruct_3d(image_path, output_dir, triposr_root=None, python_bin=None):
    triposr_root = triposr_root or DEFAULT_TRIPOSR_ROOT
    python_bin = python_bin or DEFAULT_TRIPOSR_PYTHON

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        python_bin,
        "run.py",
        str(image_path),
        "--output-dir", str(output_dir),
        "--model-save-format", "glb",
        "--foreground-ratio", "0.95",
        "--mc-resolution", "192",
        "--chunk-size", "4096"
    ], cwd=str(triposr_root), check=True)

    model_path = output_dir / "0" / "mesh.glb"
    return str(model_path)
