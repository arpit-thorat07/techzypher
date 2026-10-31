"""
Sketch2Shape AI - Remote Colab GPU Worker API Service.
Runs inside Google Colab (T4 GPU runtime) to execute:
Sketch Preprocessing -> Stable Diffusion 1.5 + ControlNet Scribble -> TripoSR -> Mesh Validation -> GLB.

Exposes:
- GET  /health: Worker readiness & GPU status
- POST /generate: Authenticated 3D generation endpoint returning the validated GLB file
"""
import asyncio
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import generate_3d, generate_3d_from_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")

# Environment configuration
WORKER_TOKEN = os.environ.get("ML_WORKER_TOKEN", "").strip()
WORKER_OUTPUTS_DIR = Path(os.environ.get("WORKER_OUTPUTS_DIR", str(PROJECT_ROOT / "worker_outputs")))
WORKER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# GPU Concurrency Lock: Prevents simultaneous GPU inference requests to prevent OOM
gpu_lock = asyncio.Lock()

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
    Authenticate incoming request against ML_WORKER_TOKEN if configured.
    Accepts either 'Authorization: Bearer <token>' or 'X-Worker-Token: <token>'.
    """
    if not WORKER_TOKEN:
        # If no token is configured on worker, allow requests (development mode)
        return

    extracted_token = None
    if authorization and authorization.startswith("Bearer "):
        extracted_token = authorization[7:].strip()
    elif x_worker_token:
        extracted_token = x_worker_token.strip()

    if not extracted_token or extracted_token != WORKER_TOKEN:
        logger.warning("Rejected request with invalid or missing worker token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing worker authentication token.",
        )


@app.get("/health", tags=["Status"])
async def health_check():
    """Worker readiness and GPU health check endpoint."""
    gpu = get_gpu_info()
    return {
        "status": "online",
        "service": "sketch2shape-colab-worker",
        "gpu": gpu,
        "is_busy": gpu_lock.locked(),
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
    2. Enforce single-request GPU lock.
    3. Run sketch preprocessing / SD 1.5 + ControlNet / TripoSR / mesh validation.
    4. Return binary GLB model.
    """
    # 1. Authenticate token
    verify_auth_token(authorization=authorization, x_worker_token=x_worker_token)

    # 2. Concurrency protection: Reject if GPU is busy
    if gpu_lock.locked():
        logger.warning("GPU is currently busy. Rejecting concurrent generation request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU worker is currently busy with another generation task. Please retry shortly.",
        )

    # 3. Acquire GPU lock for inference
    async with gpu_lock:
        job_id = uuid.uuid4().hex[:10]
        job_dir = WORKER_OUTPUTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{job_id}] Received generation request (mode={mode}, prompt='{prompt}')")

        # Save uploaded image to job directory
        input_extension = Path(file.filename or "input.png").suffix.lower()
        if input_extension not in {".png", ".jpg", ".jpeg", ".webp"}:
            input_extension = ".png"

        input_path = job_dir / f"input{input_extension}"
        try:
            with input_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"[{job_id}] Failed to save upload file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded image: {str(e)}",
            )
        finally:
            await file.close()

        # 4. Invoke notebook-tested ML pipeline
        try:
            clean_mode = (mode or "draw").lower()
            clean_prompt = (prompt or "").strip()

            # Execute pipeline synchronously in threadpool to keep event loop alive
            if clean_mode == "upload":
                logger.info(f"[{job_id}] Running image-conditioned 3D pipeline...")
                model_path = await asyncio.to_thread(
                    generate_3d_from_image,
                    str(input_path),
                    clean_prompt,
                    str(job_dir),
                )
            else:
                logger.info(f"[{job_id}] Running sketch ControlNet 3D pipeline...")
                model_path = await asyncio.to_thread(
                    generate_3d,
                    str(input_path),
                    clean_prompt,
                    str(job_dir),
                )

            model_file = Path(model_path)
            if not model_file.is_file() or model_file.stat().st_size == 0:
                raise RuntimeError("Inference finished but model.glb was not found or is empty.")

            logger.info(f"[{job_id}] Successfully generated GLB ({model_file.stat().st_size} bytes)")

            # Return real, validated binary GLB
            return FileResponse(
                path=str(model_file),
                media_type="model/gltf-binary",
                filename=f"{job_id}.glb",
            )

        except ValueError as val_err:
            logger.warning(f"[{job_id}] Validation error during preprocessing/mesh export: {val_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )
        except Exception as err:
            logger.error(f"[{job_id}] Generation pipeline failed: {err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference pipeline failed: {str(err)}",
            )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("worker:app", host="0.0.0.0", port=port, reload=False)
