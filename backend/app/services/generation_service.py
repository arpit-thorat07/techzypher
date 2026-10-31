from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import trimesh

from app.core.config import settings
from app.schemas.generation import JobStatus


class GenerationService:
    """
    Service managing 3D generation job records and model retrieval.
    Prepared for future ML inference and mesh reconstruction pipeline hooks.
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

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job record by job_id, checking outputs directory for completed models."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        # Check if an actual output model file exists in outputs directory
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
    def inspect_mesh_file(model_path: Path) -> Dict[str, Any]:
        """
        Helper using trimesh to inspect generated 3D geometry.
        Prepared for the future reconstruction pipeline.
        """
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

