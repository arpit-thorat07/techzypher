"""
Sketch2Shape AI - Remote Colab GPU Worker API Service.
Runs inside Google Colab (T4 GPU runtime) to execute:
Sketch Preprocessing -> Stable Diffusion 1.5 + ControlNet Scribble -> TripoSR -> Mesh Validation -> GLB.

Exposes:
- GET  /health: Worker readiness & GPU status
- POST /generate: Authenticated 3D generation endpoint returning the validated GLB file
"""
import asyncio
import contextlib
import hmac
import io
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Optional
import uuid

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# 1. Dynamically discover PROJECT_ROOT containing ml/
def find_project_root() -> Path:
    env_candidate = os.environ.get("PROJECT_ROOT")
    if env_candidate:
        p = Path(env_candidate).resolve()
        if (p / "ml").is_dir() and (p / "ml" / "pipeline.py").is_file():
            return p

    file_parent = Path(__file__).resolve().parent
    if (file_parent / "ml").is_dir() and (file_parent / "ml" / "pipeline.py").is_file():
        return file_parent

    cwd = Path.cwd().resolve()
    if (cwd / "ml").is_dir() and (cwd / "ml" / "pipeline.py").is_file():
        return cwd

    for loc in [
        Path("/content/techzypher"),
        Path("/content/techzypher-sketch2shape-ai"),
        Path("/content/Sketch2Shape-AI"),
        Path("/content"),
    ]:
        if (loc / "ml").is_dir() and (loc / "ml" / "pipeline.py").is_file():
            return loc

    return file_parent


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from ml.pipeline import generate_3d, generate_3d_from_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")

# 2. Worker Authentication Configuration
WORKER_TOKEN = os.environ.get("ML_WORKER_TOKEN", "").strip()
ALLOW_INSECURE_DEV = os.environ.get("ALLOW_INSECURE_DEV", "false").lower() in ("true", "1")

# Fail startup immediately in production if token is missing
if not WORKER_TOKEN and not ALLOW_INSECURE_DEV:
    error_msg = (
        "FATAL: ML_WORKER_TOKEN environment variable is required to run the worker in production. "
        "Worker startup aborted. Set ML_WORKER_TOKEN to a secure secret or Colab secret."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# 3. Upload & Timeout Configurations
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))  # 25 MB default
WORKER_TIMEOUT_SECONDS = float(os.environ.get("WORKER_TIMEOUT_SECONDS", "300.0"))  # 300s default
WORKER_OUTPUTS_DIR = Path(os.environ.get("WORKER_OUTPUTS_DIR", str(PROJECT_ROOT / "worker_outputs")))
WORKER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# 4. Strict Non-Blocking GPU Concurrency Lock
class GPULock:
    """
    Atomic non-blocking GPU concurrency lock.
    Eliminates check-then-acquire race conditions.
    Any second simultaneous request is immediately rejected with HTTP 503
    without waiting or queuing for GPU resources.
    """
    def __init__(self) -> None:
        self._locked = False
        self._sync_lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Atomically test and set lock state."""
        with self._sync_lock:
            if self._locked:
                return False
            self._locked = True
            return True

    def release(self) -> None:
        """Release the lock state."""
        with self._sync_lock:
            self._locked = False

    @property
    def is_locked(self) -> bool:
        with self._sync_lock:
            return self._locked

    @contextlib.asynccontextmanager
    async def acquire_or_503(self):
        """Context manager that acquires the lock atomically or immediately raises HTTP 503."""
        if not self.try_acquire():
            logger.warning("GPU is currently busy. Rejecting concurrent generation request with HTTP 503.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU worker is currently busy with another generation task. Please retry shortly.",
            )
        try:
            yield
        finally:
            self.release()


gpu_lock = GPULock()

app = FastAPI(
    title="Sketch2Shape AI - Remote Colab GPU Worker",
    version="1.0.0",
    description="GPU worker service executing ControlNet + TripoSR 3D generation pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_gpu_info() -> dict:
    """Detect available CUDA GPU hardware without crashing on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
            }
    except Exception as e:
        logger.warning(f"Could not query torch cuda: {e}")
    return {"available": False, "device_name": "None", "device_count": 0}


def verify_auth_token(
    authorization: Optional[str] = Header(None),
    x_worker_token: Optional[str] = Header(None),
) -> None:
    """
    Authenticate incoming request against ML_WORKER_TOKEN using constant-time comparison.
    Never logs or exposes the token value.
    """
    if not WORKER_TOKEN:
        if ALLOW_INSECURE_DEV:
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Worker token is required.",
        )

    extracted_token = None
    if authorization and authorization.startswith("Bearer "):
        extracted_token = authorization[7:].strip()
    elif x_worker_token:
        extracted_token = x_worker_token.strip()

    if not extracted_token or not hmac.compare_digest(extracted_token, WORKER_TOKEN):
        logger.warning("Rejected request with invalid or missing worker authentication token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing worker authentication token.",
        )


def validate_glb_mesh(glb_path: Path) -> None:
    """
    Validate that the generated 3D model is a valid, structurally sound GLB file:
    1. File exists and has at least 12 bytes.
    2. Begins with glTF magic header (b'glTF').
    3. Can be loaded by trimesh.
    4. Contains non-empty geometry (total vertices > 0 and faces > 0).
    5. Has finite numerical bounds.
    6. Has strictly positive 3D extents across all axes.
    """
    if not glb_path.is_file():
        raise RuntimeError(f"Generated GLB file not found: {glb_path}")

    file_size = glb_path.stat().st_size
    if file_size < 12:
        raise RuntimeError(f"Generated GLB file is truncated ({file_size} bytes).")

    with glb_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            raise RuntimeError(f"Invalid GLB magic header: {magic!r}. Expected b'glTF'.")

    try:
        import trimesh
        import numpy as np

        loaded = trimesh.load(str(glb_path), file_type="glb", force="scene")
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise RuntimeError("Generated GLB scene contains no geometry nodes.")
            total_verts = sum(len(m.vertices) for m in loaded.geometry.values() if hasattr(m, "vertices"))
            total_faces = sum(len(m.faces) for m in loaded.geometry.values() if hasattr(m, "faces"))
            bounds = loaded.bounds
            extents = loaded.extents
        elif isinstance(loaded, trimesh.Trimesh):
            total_verts = len(loaded.vertices)
            total_faces = len(loaded.faces)
            bounds = loaded.bounds
            extents = loaded.extents
        else:
            raise RuntimeError(f"Unexpected trimesh load result type: {type(loaded)}")

        if total_verts == 0:
            raise RuntimeError("Generated GLB model contains 0 vertices.")
        if total_faces == 0:
            raise RuntimeError("Generated GLB model contains 0 faces.")

        if bounds is None or not np.all(np.isfinite(bounds)):
            raise RuntimeError("Generated GLB model has non-finite or NaN bounds.")

        if extents is None or not np.all(extents > 1e-6):
            raise RuntimeError(f"Generated GLB model extents must be positive across all axes, got: {extents}")

    except Exception as e:
        raise RuntimeError(f"GLB mesh validation failed: {str(e)}")


@app.get("/health", tags=["Status"])
async def health_check():
    """Worker readiness and GPU health check endpoint."""
    gpu = get_gpu_info()
    return {
        "status": "online",
        "service": "sketch2shape-colab-worker",
        "gpu": gpu,
        "is_busy": gpu_lock.is_locked,
        "auth_enabled": bool(WORKER_TOKEN),
    }


@app.post("/generate", tags=["Inference"])
async def generate_model(
    file: UploadFile = File(..., description="Uploaded sketch or colored image file"),
    prompt: Optional[str] = Form("", description="Optional text prompt specifying materials and details"),
    mode: Optional[str] = Form("draw", description="Input mode: 'draw' (sketch) or 'upload' (colored image)"),
    authorization: Optional[str] = Header(None),
    x_worker_token: Optional[str] = Header(None),
):
    """
    Execute end-to-end 3D model generation on GPU:
    1. Authenticate worker token.
    2. Atomically acquire single-request GPU lock (HTTP 503 if busy).
    3. Validate input format, dimensions, size, and mode.
    4. Run ML pipeline with configurable timeout.
    5. Validate generated GLB with trimesh.
    6. Return binary GLB model.
    """
    # 1. Authenticate token
    verify_auth_token(authorization=authorization, x_worker_token=x_worker_token)

    # 2. Validate mode strictly (reject unsupported modes with HTTP 400)
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input mode is required ('draw' or 'upload').",
        )
    clean_mode = mode.strip().lower()
    if clean_mode not in {"draw", "upload"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported mode '{mode}'. Mode must be either 'draw' (sketch) or 'upload' (colored image).",
        )

    # 3. Atomically acquire GPU lock (HTTP 503 immediately if busy)
    async with gpu_lock.acquire_or_503():
        job_id = uuid.uuid4().hex[:10]
        job_dir = WORKER_OUTPUTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{job_id}] Received generation request (mode={clean_mode})")

        # 4. Read uploaded file with configurable size limit
        try:
            contents = await file.read()
            if not contents or len(contents) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded image file is empty (0 bytes).",
                )
            if len(contents) > MAX_UPLOAD_SIZE_BYTES:
                max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Uploaded image exceeds maximum permitted size of {max_mb:.1f} MB.",
                )

            # Validate actual image contents with Pillow
            try:
                with Image.open(io.BytesIO(contents)) as pil_img:
                    pil_img.verify()
                # Reopen to inspect format and dimensions
                with Image.open(io.BytesIO(contents)) as pil_img:
                    img_format = (pil_img.format or "").upper()
                    if img_format not in {"PNG", "JPEG", "JPG", "WEBP"}:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported image format '{img_format}'. Only PNG, JPEG, and WebP are allowed.",
                        )
                    w, h = pil_img.size
                    if w <= 0 or h <= 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Image dimensions must be greater than zero.",
                        )
                    target_ext = ".png" if img_format == "PNG" else ".jpg" if img_format in ("JPEG", "JPG") else ".webp"
            except HTTPException:
                raise
            except Exception as img_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Uploaded file is not a valid or readable image: {str(img_err)}",
                )

            input_path = job_dir / f"input{target_ext}"
            input_path.write_bytes(contents)

        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except Exception as read_err:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.error(f"[{job_id}] Failed to process uploaded image: {read_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded image: {str(read_err)}",
            )
        finally:
            await file.close()

        # 5. Invoke notebook-tested ML pipeline with configurable timeout
        try:
            clean_prompt = (prompt or "").strip()

            if clean_mode == "upload":
                logger.info(f"[{job_id}] Running image-conditioned 3D pipeline...")
                target_fn = generate_3d_from_image
            else:
                logger.info(f"[{job_id}] Running sketch ControlNet 3D pipeline...")
                target_fn = generate_3d

            model_path = await asyncio.wait_for(
                asyncio.to_thread(
                    target_fn,
                    str(input_path),
                    clean_prompt,
                    str(job_dir),
                ),
                timeout=WORKER_TIMEOUT_SECONDS,
            )

            model_file = Path(model_path)

            # 6. Validate generated GLB with trimesh before returning
            validate_glb_mesh(model_file)

            logger.info(f"[{job_id}] Successfully generated and validated GLB ({model_file.stat().st_size} bytes)")

            return FileResponse(
                path=str(model_file),
                media_type="model/gltf-binary",
                filename=f"{job_id}.glb",
            )

        except asyncio.TimeoutError:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.error(f"[{job_id}] Inference timed out after {WORKER_TIMEOUT_SECONDS} seconds")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Inference timed out after {WORKER_TIMEOUT_SECONDS:.0f} seconds.",
            )
        except ValueError as val_err:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.warning(f"[{job_id}] Validation error during generation: {val_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )
        except Exception as err:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.error(f"[{job_id}] Generation pipeline failed: {err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference pipeline failed: {str(err)}",
            )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("worker:app", host="0.0.0.0", port=port, reload=False)
