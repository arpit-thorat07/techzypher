import os
from pathlib import Path
import subprocess

DEFAULT_TRIPOSR_ROOT = os.environ.get("TRIPOSR_ROOT", "/content/TripoSR")
DEFAULT_TRIPOSR_PYTHON = os.environ.get("TRIPOSR_PYTHON", "/content/triposr-env/bin/python")
SUBPROCESS_TIMEOUT_SECONDS = float(os.environ.get("ML_SUBPROCESS_TIMEOUT_SECONDS", "300.0"))


def reconstruct_3d(image_path, output_dir, triposr_root=None, python_bin=None):
    """
    Run TripoSR 3D reconstruction on the input image.
    Outputs GLB to output_dir/0/mesh.glb.
    """
    triposr_root = Path(triposr_root or DEFAULT_TRIPOSR_ROOT)
    python_bin = Path(python_bin or DEFAULT_TRIPOSR_PYTHON)

    if not triposr_root.is_dir() or not (triposr_root / "run.py").is_file():
        raise RuntimeError(
            f"TripoSR root directory invalid: {triposr_root}. "
            "Ensure TripoSR is cloned and contains run.py."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        str(python_bin),
        "run.py",
        str(image_path),
        "--output-dir", str(output_dir),
        "--model-save-format", "glb",
        "--foreground-ratio", "0.95",
        "--mc-resolution", "192",
        "--chunk-size", "4096"
    ], cwd=str(triposr_root), check=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)

    model_path = output_dir / "0" / "mesh.glb"
    if not model_path.is_file():
        raise RuntimeError(f"TripoSR completed but expected mesh not found at {model_path}")

    return str(model_path)
