import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import trimesh

from app.core.config import settings
from app.schemas.generation import JobStatus
from app.services.worker_client import worker_client, MLWorkerClient

logger = logging.getLogger("generation_service")


class GenerationService:
    """
    Service managing 3D generation job records, remote GPU worker orchestration,
    mesh validation, and 3D asset retrieval.
    """

    def __init__(self) -> None:
        # In-memory registry for generation jobs
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(
        self,
        job_id: str,
        input_path: Path,
        prompt: Optional[str] = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new generation job with initial queued status."""
        job_record = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "input_path": str(input_path),
            "prompt": prompt or "",
            "created_at": datetime.now(timezone.utc),
            "metadata": metadata or {},
            "model_path": None,
            "error": None,
        }
        self._jobs[job_id] = job_record
        return job_record

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update lifecycle status and error/model details of an existing job."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        job["status"] = status
        if error is not None:
            job["error"] = error
        elif status in (JobStatus.COMPLETED, JobStatus.PROCESSING):
            job["error"] = None

        if model_path is not None:
            job["model_path"] = str(model_path)

        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job record by job_id, checking outputs directory for completed models."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        # Check if an actual output model file exists in outputs directory (only if not failed)
        if job.get("status") != JobStatus.FAILED:
            model_file = self.get_model_file(job_id)
            if model_file:
                job["status"] = JobStatus.COMPLETED
                job["model_path"] = str(model_file)

        return job

    def get_model_file(self, job_id: str) -> Optional[Path]:
        """
        Check if a real generated 3D model asset exists for job_id.
        Supported formats: GLB, OBJ.
        """
        settings.init_directories()

        # Check for GLB or OBJ model outputs
        for ext in [".glb", ".gltf", ".obj"]:
            candidate = settings.OUTPUTS_DIR / f"{job_id}{ext}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

        return None

    @staticmethod
    def validate_glb(model_path: Path) -> bool:
        """
        Validate that the generated model file is a real, non-empty, and structurally sound GLB:
        - File exists and has positive size
        - Binary GLB magic header (b'glTF')
        - Valid non-empty geometry
        - Finite bounds and positive extents
        """
        if not model_path.is_file() or model_path.stat().st_size < 100:
            return False

        # Verify binary GLB header magic bytes (0x46546C67 -> 'glTF')
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                if magic != b"glTF":
                    logger.warning(f"File {model_path} lacks binary glTF header magic.")
                    return False
        except Exception as e:
            logger.warning(f"Failed to read header of {model_path}: {e}")
            return False

        try:
            loaded = trimesh.load(str(model_path), file_type="glb", force="scene")
            if isinstance(loaded, trimesh.Scene):
                if not loaded.geometry:
                    logger.warning(f"GLB scene in {model_path} contains no geometry.")
                    return False
                total_vertices = sum(len(m.vertices) for m in loaded.geometry.values() if hasattr(m, "vertices"))
                total_faces = sum(len(m.faces) for m in loaded.geometry.values() if hasattr(m, "faces"))
                bounds = loaded.bounds
                extents = loaded.extents
            elif isinstance(loaded, trimesh.Trimesh):
                total_vertices = len(loaded.vertices)
                total_faces = len(loaded.faces)
                bounds = loaded.bounds
                extents = loaded.extents
            else:
                logger.warning(f"Unexpected trimesh load result type: {type(loaded)}")
                return False

            if total_vertices == 0 or total_faces == 0:
                logger.warning(f"GLB in {model_path} contains empty geometry ({total_vertices} verts, {total_faces} faces).")
                return False

            if bounds is None or not np.all(np.isfinite(bounds)):
                logger.warning(f"GLB in {model_path} has non-finite bounds.")
                return False

            if extents is None or not np.all(extents > 1e-6):
                logger.warning(f"GLB in {model_path} has non-positive extents: {extents}.")
                return False

            return True
        except Exception as e:
            logger.warning(f"GLB geometry validation failed for {model_path}: {e}")
            return False

    async def process_job_with_worker(
        self,
        job_id: str,
        input_path: Path,
        prompt: str,
        mode: str,
        client: Optional[MLWorkerClient] = None,
    ) -> None:
        """
        Orchestrate remote GPU inference in background:
        1. Set status to processing.
        2. Forward image to remote Colab worker.
        3. Save returned GLB to backend/outputs/{job_id}.glb.
        4. Validate 3D mesh integrity.
        5. Set status to completed (or failed if any step fails).
        """
        active_client = client or worker_client
        target_output_glb = settings.OUTPUTS_DIR / f"{job_id}.glb"

        try:
            # Transition to processing status
            self.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"[{job_id}] Job transitioned to PROCESSING. Forwarding to worker...")

            # Forward to remote worker
            await active_client.forward_generation(
                image_path=input_path,
                prompt=prompt,
                mode=mode,
                output_glb_path=target_output_glb,
            )

            # Validate the received GLB file
            if not self.validate_glb(target_output_glb):
                if target_output_glb.exists():
                    target_output_glb.unlink()
                raise ValueError("Generated 3D asset failed mesh structural integrity validation.")

            # Mark as completed
            self.update_job_status(job_id, JobStatus.COMPLETED, model_path=str(target_output_glb))
            logger.info(f"[{job_id}] Inference succeeded and validated! Job marked as COMPLETED.")

        except Exception as err:
            logger.error(f"[{job_id}] Remote inference job failed: {err}")
            # Ensure failed job never leaves invalid or partial GLB files behind
            if target_output_glb.exists():
                try:
                    target_output_glb.unlink()
                except Exception:
                    pass
            self.update_job_status(job_id, JobStatus.FAILED, error=str(err))

    @staticmethod
    def inspect_mesh_file(model_path: Path) -> Dict[str, Any]:
        """Helper using trimesh to inspect generated 3D geometry."""
        try:
            mesh = trimesh.load(str(model_path), force="mesh")
            return {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "is_watertight": getattr(mesh, "is_watertight", False),
                "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
            }
        except Exception as e:
            return {"error": f"Failed to inspect mesh: {str(e)}"}


# Global singleton service instance
generation_service = GenerationService()

