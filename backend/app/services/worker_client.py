"""
Sketch2Shape AI - Remote ML Worker Client.
Coordinates HTTP communication between the local FastAPI backend and the remote GPU worker (e.g. running on Google Colab).
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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
        """Construct request headers, injecting bearer auth token when configured."""
        headers = {}
        if self.worker_token:
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
                    return {"configured": True, "reachable": True, **data}
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

    async def forward_generation(
        self,
        image_path: Path,
        prompt: str,
        mode: str,
        output_glb_path: Path,
    ) -> Path:
        """
        Forward the input image, prompt, and mode to the remote GPU worker,
        streaming/saving the returned validated GLB to output_glb_path.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Remote ML worker URL is not configured. Set ML_WORKER_URL in backend environment or .env."
            )

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image file does not exist: {image_path}")

        target_url = f"{self.worker_url}/generate"
        headers = self._get_headers()

        logger.info(
            f"Forwarding generation task to remote worker at {self.worker_url} "
            f"(mode={mode}, prompt='{prompt[:40]}...')"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(image_path, "rb") as f:
                    files = {
                        "file": (image_path.name, f, "image/png"),
                    }
                    data = {
                        "prompt": prompt or "",
                        "mode": mode or "draw",
                    }

                    response = await client.post(
                        target_url,
                        headers=headers,
                        files=files,
                        data=data,
                    )

            # Handle HTTP response codes
            if response.status_code == 200:
                content = response.content
                if not content or len(content) < 100:
                    raise RuntimeError("Remote worker returned an empty or invalid GLB payload.")

                # Ensure output directory exists and write GLB file
                output_glb_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_glb_path, "wb") as out_f:
                    out_f.write(content)

                logger.info(
                    f"Successfully received generated GLB from worker: "
                    f"{output_glb_path.name} ({len(content)} bytes)"
                )
                return output_glb_path

            # Handle error status codes
            error_detail = "Inference failed"
            try:
                err_json = response.json()
                error_detail = err_json.get("detail", response.text[:300])
            except Exception:
                error_detail = response.text[:300] or f"HTTP {response.status_code}"

            if response.status_code == 401 or response.status_code == 403:
                raise RuntimeError(
                    "Remote ML worker authentication failed. Verify ML_WORKER_TOKEN in backend environment."
                )
            elif response.status_code == 503:
                raise RuntimeError(
                    f"Remote ML worker is busy processing another GPU task: {error_detail}"
                )
            elif response.status_code == 400:
                raise ValueError(
                    f"Remote ML worker rejected input: {error_detail}"
                )
            else:
                raise RuntimeError(
                    f"Remote ML worker error ({response.status_code}): {error_detail}"
                )

        except httpx.ConnectError:
            raise RuntimeError(
                f"Failed to connect to remote ML worker at {self.worker_url}. "
                "Ensure the Google Colab worker is running and the tunnel (e.g. Cloudflare/ngrok) is active."
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Remote ML worker timed out after {self.timeout} seconds while processing 3D generation."
            )
        except (RuntimeError, ValueError, FileNotFoundError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error communicating with remote ML worker: {str(e)}")


# Default worker client instance
worker_client = MLWorkerClient()
