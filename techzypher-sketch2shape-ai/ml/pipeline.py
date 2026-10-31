import os
from pathlib import Path
import subprocess
import uuid

from ml.preprocessing import preprocess_sketch
from ml.reconstruction import reconstruct_3d
from ml.mesh_utils import validate_and_export

CONTROLNET_PYTHON = os.environ.get("CONTROLNET_PYTHON", "/content/controlnet-env/bin/python")
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))


def generate_3d(sketch_path, prompt, output_dir="outputs"):
    job_id = uuid.uuid4().hex[:10]
    job_dir = Path(output_dir).resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    control_path = job_dir / "control.png"
    image_path = job_dir / "generated.png"
    model_path = job_dir / "model.glb"

    preprocess_sketch(sketch_path, control_path)

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(PROJECT_ROOT)

    subprocess.run([
        CONTROLNET_PYTHON,
        "-c",
        (
            "import sys; "
            "from ml.image_generation import generate_image; "
            "generate_image(*sys.argv[1:4])"
        ),
        str(control_path),
        prompt,
        str(image_path)
    ], cwd=str(PROJECT_ROOT), env=env, check=True)

    raw_model = reconstruct_3d(
        str(image_path),
        str(job_dir / "reconstruction")
    )

    validate_and_export(raw_model, model_path)
    return str(model_path)


def generate_3d_from_image(image_path, prompt, output_dir="outputs"):
    job_id = uuid.uuid4().hex[:10]
    job_dir = Path(output_dir).resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    conditioned_path = job_dir / "conditioned.png"
    model_path = job_dir / "model.glb"

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(PROJECT_ROOT)

    subprocess.run([
        CONTROLNET_PYTHON,
        "-c",
        (
            "import sys; "
            "from ml.image_conditioning import condition_image; "
            "condition_image(*sys.argv[1:4])"
        ),
        str(image_path),
        prompt,
        str(conditioned_path)
    ], cwd=str(PROJECT_ROOT), env=env, check=True)

    raw_model = reconstruct_3d(
        str(conditioned_path),
        str(job_dir / "reconstruction")
    )

    validate_and_export(raw_model, model_path)
    return str(model_path)
