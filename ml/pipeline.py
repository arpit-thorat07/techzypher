import os
from pathlib import Path
import subprocess
import uuid

from ml.preprocessing import preprocess_sketch
from ml.reconstruction import reconstruct_3d
from ml.mesh_utils import validate_and_export


def find_project_root() -> Path:
    """
    Locate the root directory that directly contains the `ml/` package directory.
    Never assumes `/content` unless `ml/` actually exists directly inside `/content`.
    """
    # 1. User/environment specified PROJECT_ROOT
    env_candidate = os.environ.get("PROJECT_ROOT")
    if env_candidate:
        p = Path(env_candidate).resolve()
        if (p / "ml").is_dir() and (p / "ml" / "pipeline.py").is_file():
            return p

    # 2. Parent directory of the ml package (standard layout where ml/ is in project root)
    file_parent = Path(__file__).resolve().parent  # .../ml
    root_candidate = file_parent.parent
    if (root_candidate / "ml").is_dir() and (root_candidate / "ml" / "pipeline.py").is_file():
        return root_candidate

    # 3. Current working directory
    cwd = Path.cwd().resolve()
    if (cwd / "ml").is_dir() and (cwd / "ml" / "pipeline.py").is_file():
        return cwd

    # 4. Common Colab checkout locations
    for loc in [
        Path("/content/techzypher"),
        Path("/content/techzypher-sketch2shape-ai"),
        Path("/content/Sketch2Shape-AI"),
        Path("/content"),
    ]:
        if (loc / "ml").is_dir() and (loc / "ml" / "pipeline.py").is_file():
            return loc

    return root_candidate


CONTROLNET_PYTHON = os.environ.get("CONTROLNET_PYTHON", "/content/controlnet-env/bin/python")
PROJECT_ROOT = find_project_root()
# Export discovered PROJECT_ROOT so child processes inherit it
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

SUBPROCESS_TIMEOUT_SECONDS = float(os.environ.get("ML_SUBPROCESS_TIMEOUT_SECONDS", "300.0"))


def generate_3d(sketch_path, prompt, output_dir="outputs"):
    """
    Path 1: Draw mode pipeline.
    Preprocesses sketch -> ControlNet SD 1.5 -> TripoSR -> Mesh validation & export.
    """
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
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)

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
    ], cwd=str(PROJECT_ROOT), env=env, check=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)

    raw_model = reconstruct_3d(
        str(image_path),
        str(job_dir / "reconstruction")
    )

    validate_and_export(raw_model, model_path)
    return str(model_path)


def generate_3d_from_image(image_path, prompt, output_dir="outputs"):
    """
    Path 2: Upload mode pipeline.
    Conditions uploaded colored image -> TripoSR -> Mesh validation & export.
    """
    job_id = uuid.uuid4().hex[:10]
    job_dir = Path(output_dir).resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    conditioned_path = job_dir / "conditioned.png"
    model_path = job_dir / "model.glb"

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(PROJECT_ROOT)
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)

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
    ], cwd=str(PROJECT_ROOT), env=env, check=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)

    raw_model = reconstruct_3d(
        str(conditioned_path),
        str(job_dir / "reconstruction")
    )

    validate_and_export(raw_model, model_path)
    return str(model_path)
