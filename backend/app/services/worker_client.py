"""
Sketch2Shape AI - Remote ML Worker Client.
Coordinates HTTP communication between the local FastAPI backend and the remote GPU worker (e.g. running on Google Colab).
Supports both:
1. Asynchronous job queue protocol (POST /jobs -> poll GET /jobs/{id} -> GET /jobs/{id}/bundle ZIP).
2. Direct streaming protocol (POST /generate returning GLB directly).
"""
import asyncio
import io
import logging
from pathlib import Path
import time
from typing import Optional, Dict, Any
from zipfile import ZipFile

import httpx

from app.core.config import settings

logger = logging.getLogger("worker_client")


class MLWorkerClient:
    """Async HTTP client for dispatching inference jobs to the remote Colab GPU worker."""

    def __init__(
        self,
        worker_url: Optional[str] = None,
        worker_token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        raw_url = (worker_url or settings.ML_WORKER_URL or "").strip()
        self.worker_url = raw_url.rstrip("/") if raw_url else None
        self.worker_token = (worker_token or settings.ML_WORKER_TOKEN or "").strip()
        self.timeout = timeout or settings.ML_WORKER_TIMEOUT_SECONDS

    @property
    def is_configured(self) -> bool:
        """Return True if remote worker URL is configured."""
        return bool(self.worker_url)

    def _get_headers(self) -> Dict[str, str]:
        """
        Construct request headers, injecting authentication tokens for both
        X-ML-Worker-Token and Authorization: Bearer formats.
        """
        headers = {}
        if self.worker_token:
            headers["X-ML-Worker-Token"] = self.worker_token
            headers["Authorization"] = f"Bearer {self.worker_token}"
        return headers

    async def health_check(self) -> Dict[str, Any]:
        """Query the remote worker's health endpoint to check readiness and GPU status."""
        if not self.is_configured:
            return {"configured": False, "status": "unconfigured"}

        target_url = f"{self.worker_url}/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(target_url, headers=self._get_headers())
                if res.status_code == 200:
                    data = res.json()
                    status_text = data.get("status", "online")
                    is_ready = status_text in ("ready", "online")
                    return {
                        "configured": True,
                        "reachable": True,
                        "status": status_text,
                        "ready": is_ready,
                        **data,
                    }
                return {
                    "configured": True,
                    "reachable": True,
                    "status_code": res.status_code,
                    "detail": res.text[:200],
                }
        except httpx.ConnectError:
            return {
                "configured": True,
                "reachable": False,
                "error": f"Cannot connect to worker at {self.worker_url}. Verify Colab worker and tunnel are active.",
            }
        except Exception as e:
            return {
                "configured": True,
                "reachable": False,
                "error": str(e),
            }

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """Map frontend mode names ('draw', 'upload') to worker mode names ('sketch', 'color')."""
        clean = (mode or "draw").strip().lower()
        if clean in ("draw", "sketch"):
            return "sketch"
        return "color"

    async def forward_generation(
        self,
        image_path: Path,
        prompt: str,
        mode: str,
        output_glb_path: Path,
    ) -> Path:
        """
        Forward the input image, prompt, and mode to the remote GPU worker.
        Tries the asynchronous /jobs protocol first, then falls back to /generate if unavailable.
        Saves the validated binary GLB to output_glb_path.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Remote ML worker URL is not configured. Set ML_WORKER_URL in backend environment or .env."
            )

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image file does not exist: {image_path}")

        headers = self._get_headers()
        worker_mode = self._normalize_mode(mode)

        logger.info(
            f"Forwarding generation task to remote worker at {self.worker_url} "
            f"(mode={worker_mode}, prompt='{prompt[:40]}...')"
        )

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        file_extension = image_path.suffix.lower()
        content_type = (
            "image/jpeg" if file_extension in (".jpg", ".jpeg")
            else "image/webp" if file_extension == ".webp"
            else "image/png"
        )

        # -------------------------------------------------------------
        # Protocol 1: Asynchronous Job Queue (POST /jobs -> poll /jobs/{id})
        # -------------------------------------------------------------
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                submit_res = await client.post(
                    f"{self.worker_url}/jobs",
                    headers=headers,
                    data={"prompt": prompt, "mode": worker_mode},
                    files={"image": (image_path.name, image_bytes, content_type)},
                )

            if submit_res.status_code == 202 or submit_res.status_code == 200:
                job_data = submit_res.json()
                worker_job_id = job_data.get("job_id")
                logger.info(f"Worker accepted job: {worker_job_id} (status: queued). Polling for completion...")

                # Poll worker until job completes
                return await self._poll_and_download_bundle(
                    worker_job_id=worker_job_id,
                    output_glb_path=output_glb_path,
                    headers=headers,
                )

            elif submit_res.status_code == 429:
                raise RuntimeError("Remote GPU worker queue is full (HTTP 429). Please retry shortly.")
            elif submit_res.status_code == 401 or submit_res.status_code == 403:
                raise RuntimeError("Remote ML worker authentication failed. Verify ML_WORKER_TOKEN.")
            elif submit_res.status_code != 404:
                error_detail = submit_res.text[:300]
                try:
                    error_detail = submit_res.json().get("detail", error_detail)
                except Exception:
                    pass
                raise RuntimeError(f"Remote worker rejected job submission ({submit_res.status_code}): {error_detail}")

        except httpx.ConnectError:
            raise RuntimeError(
                f"Failed to connect to remote ML worker at {self.worker_url}. "
                "Ensure the Google Colab worker is running and the tunnel (e.g. Cloudflare) is active."
            )
        except (RuntimeError, ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.warning(f"Error attempting /jobs endpoint on worker: {e}. Checking fallback...")

        # -------------------------------------------------------------
        # Protocol 2: Direct Streaming Fallback (POST /generate)
        # -------------------------------------------------------------
        logger.info(f"Attempting direct /generate streaming endpoint on {self.worker_url}...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.worker_url}/generate",
                    headers=headers,
                    files={"file": (image_path.name, image_bytes, content_type)},
                    data={"prompt": prompt, "mode": mode},
                )

            if response.status_code == 200:
                content = response.content
                if not content or len(content) < 100:
                    raise RuntimeError("Remote worker returned an empty or invalid GLB payload.")

                output_glb_path.parent.mkdir(parents=True, exist_ok=True)
                output_glb_path.write_bytes(content)
                logger.info(f"Successfully received generated GLB from worker: {output_glb_path.name}")
                return output_glb_path

            error_detail = response.text[:300]
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass

            if response.status_code == 401 or response.status_code == 403:
                raise RuntimeError("Remote ML worker authentication failed. Verify ML_WORKER_TOKEN in backend environment.")
            elif response.status_code == 503:
                raise RuntimeError(f"Remote ML worker is busy processing another GPU task: {error_detail}")
            elif response.status_code == 400 or response.status_code == 422:
                raise ValueError(f"Remote ML worker rejected input: {error_detail}")
            else:
                raise RuntimeError(f"Remote ML worker error ({response.status_code}): {error_detail}")

        except httpx.ConnectError:
            raise RuntimeError(
                f"Failed to connect to remote ML worker at {self.worker_url}. "
                "Ensure the Google Colab worker is running and the tunnel (e.g. Cloudflare/ngrok) is active."
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Remote ML worker timed out after {self.timeout} seconds while processing 3D generation."
            )

    async def _poll_and_download_bundle(
        self,
        worker_job_id: str,
        output_glb_path: Path,
        headers: Dict[str, str],
    ) -> Path:
        """
        Poll worker GET /jobs/{worker_job_id} until completed or failed,
        then download the bundle ZIP and extract model.glb.
        """
        status_url = f"{self.worker_url}/jobs/{worker_job_id}"
        deadline = time.monotonic() + self.timeout
        poll_interval = 3.0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < deadline:
                try:
                    res = await client.get(status_url, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        job_status = data.get("status", "").lower()
                        logger.info(f"Worker job {worker_job_id} status: {job_status}")

                        if job_status == "completed":
                            bundle_rel_url = data.get("bundle_url") or f"/jobs/{worker_job_id}/bundle"
                            bundle_url = f"{self.worker_url}{bundle_rel_url}"

                            logger.info(f"Downloading completed bundle from {bundle_url}...")
                            async with httpx.AsyncClient(timeout=120.0) as dl_client:
                                dl_res = await dl_client.get(bundle_url, headers=headers)
                                dl_res.raise_for_status()
                                payload = dl_res.content

                            # Handle ZIP bundle
                            if payload.startswith(b"PK\x03\x04"):
                                with ZipFile(io.BytesIO(payload)) as bundle:
                                    namelist = bundle.namelist()
                                    if "model.glb" not in namelist:
                                        raise RuntimeError("Worker bundle ZIP did not contain model.glb")

                                    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
                                    output_glb_path.write_bytes(bundle.read("model.glb"))

                                    # Also extract preview if present
                                    if "preview.png" in namelist:
                                        preview_path = output_glb_path.with_name(f"{output_glb_path.stem}_preview.png")
                                        preview_path.write_bytes(bundle.read("preview.png"))

                                logger.info(
                                    f"Extracted model.glb from bundle ({output_glb_path.stat().st_size} bytes)"
                                )
                                return output_glb_path
                            elif payload.startswith(b"glTF"):
                                output_glb_path.parent.mkdir(parents=True, exist_ok=True)
                                output_glb_path.write_bytes(payload)
                                return output_glb_path
                            else:
                                raise RuntimeError("Worker returned unexpected payload format (neither ZIP nor GLB).")

                        elif job_status == "failed":
                            err_msg = data.get("error") or "GPU inference failed on remote Colab worker."
                            raise RuntimeError(f"Remote ML worker reported failure: {err_msg}")

                    elif res.status_code == 404:
                        raise RuntimeError(f"Worker job {worker_job_id} expired or was not found on worker.")
                    elif res.status_code in (401, 403):
                        raise RuntimeError("Worker authentication failed during status polling.")

                except (RuntimeError, ValueError):
                    raise
                except Exception as poll_err:
                    logger.warning(f"Transient error polling worker job {worker_job_id}: {poll_err}")

                await asyncio.sleep(poll_interval)

        raise RuntimeError(
            f"Remote ML worker timed out after {self.timeout} seconds while processing job {worker_job_id}."
        )


# Default worker client instance
worker_client = MLWorkerClient()
