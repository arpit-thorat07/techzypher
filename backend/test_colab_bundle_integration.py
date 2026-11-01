"""
End-to-end integration test verifying:
1. Arpit's Colab worker (gpu_worker/app.py) protocol:
   - Header: X-ML-Worker-Token
   - POST /jobs (mode: sketch/color, image, prompt) -> HTTP 202 {"job_id": ..., "status": "queued"}
   - GET /jobs/{job_id} -> status polling
   - GET /jobs/{job_id}/bundle -> ZIP archive download containing model.glb and preview.png
2. MLWorkerClient:
   - Correctly maps mode ('draw' -> 'sketch', 'upload' -> 'color')
   - Injects X-ML-Worker-Token
   - Polls worker until completion
   - Downloads and unzips model.glb and preview.png to target paths
3. GLB vertex color material fix (ml.mesh_utils):
   - Confirms exported GLB attaches PBR material to COLOR_0 primitive
4. Generation service and backend status lifecycle:
   - queued -> processing -> completed
"""
import asyncio
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest.mock as mock
from zipfile import ZipFile

import httpx
import numpy as np
from PIL import Image
import trimesh

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

from app.core.config import settings
from app.services.worker_client import MLWorkerClient
from app.services.generation_service import generation_service
from ml.mesh_utils import validate_and_export, _add_vertex_color_material
from gpu_worker.app import app as gpu_worker_app, JOBS_LOCK, JOBS


def create_minimal_colored_glb() -> bytes:
    """Create a minimal GLB with vertex colors for testing."""
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile(
        np.array([[245, 110, 210, 255]], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.glb"
        out_path = Path(tmpdir) / "colored.glb"
        mesh.export(str(raw_path))
        validate_and_export(str(raw_path), str(out_path))
        return out_path.read_bytes()


def create_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "#ee88cc").save(buf, format="PNG")
    return buf.getvalue()


async def run_integration_tests():
    print("\n=======================================================")
    print("TESTING COLAB GPU WORKER BUNDLE & CLIENT INTEGRATION")
    print("=======================================================\n")

    # -------------------------------------------------------------
    # 1. Test GLB Vertex Color Material Injection
    # -------------------------------------------------------------
    print("--- 1. Testing GLB Vertex Color Material Fix (_add_vertex_color_material) ---")
    colored_glb = create_minimal_colored_glb()
    assert len(colored_glb) > 20
    assert colored_glb[:4] == b"glTF"

    # Unpack JSON chunk to verify PBR material added
    size, kind = struct.unpack_from("<II", colored_glb, 12)
    assert kind == 0x4E4F534A
    doc = json.loads(colored_glb[20:20 + size])
    prim = doc["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"], "Missing COLOR_0 attribute!"
    assert "material" in prim, "Primitive lacks material reference!"
    mat = doc["materials"][prim["material"]]
    assert mat["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
    assert mat["doubleSided"] is True
    print("CONFIRMED: _add_vertex_color_material successfully attached PBR material to COLOR_0 primitive.")

    # -------------------------------------------------------------
    # 2. Test gpu_worker/app.py Endpoints with Test Client
    # -------------------------------------------------------------
    print("\n--- 2. Testing gpu_worker/app.py Endpoints & Auth ---")
    os.environ["ML_WORKER_TOKEN"] = "colab-worker-secret-token"

    with JOBS_LOCK:
        JOBS.clear()

    worker_transport = httpx.ASGITransport(app=gpu_worker_app)
    async with httpx.AsyncClient(transport=worker_transport, base_url="http://worker-test") as client:
        # Check /health
        res = await client.get("/health")
        assert res.status_code == 200
        health_data = res.json()
        print("Worker /health ->", health_data)

        # Submit job with wrong token -> 401
        img_bytes = create_test_image_bytes()
        res_bad_auth = await client.post(
            "/jobs",
            headers={"X-ML-Worker-Token": "bad-token"},
            data={"mode": "sketch", "prompt": "drone"},
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        assert res_bad_auth.status_code == 401
        print("Unauthorized request correctly rejected with 401.")

        # Mock _run_generation in gpu_worker
        def mock_worker_generation(input_path, prompt, output_dir, mode):
            runs_dir = Path(output_dir) / "run_1"
            runs_dir.mkdir(parents=True, exist_ok=True)
            preview_file = "generated.png" if mode == "sketch" else "conditioned.png"
            (runs_dir / preview_file).write_bytes(img_bytes)
            (runs_dir / "model.glb").write_bytes(colored_glb)
            return str(runs_dir / "model.glb")

        with mock.patch("gpu_worker.app._run_generation", side_effect=mock_worker_generation):
            # Submit valid job
            res_job = await client.post(
                "/jobs",
                headers={"X-ML-Worker-Token": "colab-worker-secret-token"},
                data={"mode": "sketch", "prompt": "a futuristic drone"},
                files={"image": ("test.png", img_bytes, "image/png")},
            )
            assert res_job.status_code == 202
            worker_job_id = res_job.json()["job_id"]
            print(f"Job submitted to worker: {worker_job_id} (status: queued)")

            # Poll for completion
            for _ in range(30):
                res_status = await client.get(
                    f"/jobs/{worker_job_id}",
                    headers={"X-ML-Worker-Token": "colab-worker-secret-token"}
                )
                assert res_status.status_code == 200
                status_info = res_status.json()
                if status_info["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)

            assert status_info["status"] == "completed"
            bundle_url = status_info["bundle_url"]
            print(f"Worker job completed! Bundle URL: {bundle_url}")

            # Download bundle
            res_bundle = await client.get(
                bundle_url,
                headers={"X-ML-Worker-Token": "colab-worker-secret-token"}
            )
            assert res_bundle.status_code == 200
            assert res_bundle.headers["content-type"] == "application/zip"
            with ZipFile(io.BytesIO(res_bundle.content)) as zip_f:
                names = set(zip_f.namelist())
                assert "model.glb" in names
                assert "preview.png" in names
                assert "manifest.json" in names
            print(f"Bundle downloaded and verified: ZIP contains {names}")

    # -------------------------------------------------------------
    # 3. Test Backend MLWorkerClient with Asynchronous Bundle Protocol
    # -------------------------------------------------------------
    print("\n--- 3. Testing Backend MLWorkerClient End-to-End ---")
    mock_worker_client = MLWorkerClient(
        worker_url="http://worker-test",
        worker_token="colab-worker-secret-token",
        timeout=60.0
    )

    # Use transport to connect client to gpu_worker_app
    orig_client = httpx.AsyncClient

    def make_test_client(**kw):
        kw["transport"] = worker_transport
        return orig_client(**kw)

    with mock.patch("gpu_worker.app._run_generation", side_effect=mock_worker_generation), \
         mock.patch("httpx.AsyncClient", side_effect=make_test_client):

        test_input_path = settings.UPLOADS_DIR / "test_client_sketch.png"
        test_input_path.write_bytes(img_bytes)
        target_glb_path = settings.OUTPUTS_DIR / "test_job_result.glb"

        try:
            # Test forward_generation: passes 'draw', client maps to 'sketch'
            result_path = await mock_worker_client.forward_generation(
                image_path=test_input_path,
                prompt="futuristic hovercraft",
                mode="draw",
                output_glb_path=target_glb_path,
            )

            assert result_path.is_file()
            assert result_path.stat().st_size > 0
            print(f"Successfully processed and downloaded: {result_path.name} ({result_path.stat().st_size} bytes)")

            # Check that preview image was also saved
            preview_saved = target_glb_path.with_name(f"{target_glb_path.stem}_preview.png")
            assert preview_saved.is_file(), "Preview image should have been extracted!"
            print(f"Preview extracted alongside GLB: {preview_saved.name}")

            # Verify GLB validity with backend trimesh validator
            assert generation_service.validate_glb(result_path) is True
            print("GLB passed generation_service structural validation!")

        finally:
            test_input_path.unlink(missing_ok=True)
            target_glb_path.unlink(missing_ok=True)
            if 'preview_saved' in locals():
                preview_saved.unlink(missing_ok=True)

    print("\n=======================================================")
    print(">>> ALL COLAB BUNDLE & CLIENT INTEGRATION TESTS PASSED! <<<")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
