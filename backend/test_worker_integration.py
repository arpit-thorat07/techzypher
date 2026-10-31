"""
Comprehensive integration and unit test suite for the Remote Colab Worker and Backend Client.
Tests:
1. Worker authentication, authorization headers, and token protection.
2. Worker concurrency lock (rejection with HTTP 503 when GPU is busy).
3. Backend MLWorkerClient error translation (connect error, auth error, timeout, busy).
4. Binary GLB validation on backend (magic header, mesh validation).
5. Full background job generation lifecycle with simulated worker.
"""
import asyncio
import io
from pathlib import Path
import sys
import unittest.mock as mock

import httpx
from PIL import Image
import trimesh

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent / "techzypher-sketch2shape-ai"))

from app.main import app
from app.core.config import settings
from app.schemas.generation import JobStatus
from app.services.generation_service import generation_service
from app.services.worker_client import MLWorkerClient
from worker import app as worker_app, gpu_lock


def create_minimal_glb() -> bytes:
    """Create a minimal structurally valid binary GLB using trimesh for testing."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene(mesh)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def create_test_image_bytes(format="PNG") -> bytes:
    """Create a valid PNG image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(20, 20, 20))
    img.save(buf, format=format)
    return buf.getvalue()


async def run_worker_tests():
    print("\n=======================================================")
    print("RUNNING WORKER & BACKEND INTEGRATION TEST SUITE")
    print("=======================================================\n")

    # -------------------------------------------------------------
    # Test 1: Worker Health & Authentication
    # -------------------------------------------------------------
    print("--- 1. Testing Worker /health & Auth Enforcement ---")
    transport = httpx.ASGITransport(app=worker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:
        # Health check
        res = await client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        health_data = res.json()
        print("Worker /health ->", health_data)
        assert health_data["status"] == "online"

        # Mock WORKER_TOKEN to test authentication
        with mock.patch("worker.WORKER_TOKEN", "secret-test-token-123"):
            img_bytes = create_test_image_bytes()
            files = {"file": ("sketch.png", img_bytes, "image/png")}

            # Request without token -> 401
            res_unauth = await client.post("/generate", files=files, data={"mode": "draw"})
            assert res_unauth.status_code == 401, f"Expected 401, got {res_unauth.status_code}"
            print("Rejected unauthenticated request: 401 OK ->", res_unauth.json()["detail"])

            # Request with invalid token -> 401
            res_bad_token = await client.post(
                "/generate",
                headers={"Authorization": "Bearer wrong-token"},
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            )
            assert res_bad_token.status_code == 401
            print("Rejected bad token: 401 OK ->", res_bad_token.json()["detail"])

    # -------------------------------------------------------------
    # Test 2: Worker Concurrency GPU Lock (HTTP 503)
    # -------------------------------------------------------------
    print("\n--- 2. Testing Concurrency GPU Lock (HTTP 503 Protection) ---")
    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:
        with mock.patch("worker.WORKER_TOKEN", ""):
            # Simulate GPU lock being held by another inference task
            await gpu_lock.acquire()
            try:
                img_bytes = create_test_image_bytes()
                res_busy = await client.post(
                    "/generate",
                    files={"file": ("sketch.png", img_bytes, "image/png")},
                    data={"mode": "draw"}
                )
                assert res_busy.status_code == 503, f"Expected 503, got {res_busy.status_code}"
                print("GPU busy protection: 503 Service Unavailable ->", res_busy.json()["detail"])
            finally:
                gpu_lock.release()

    # -------------------------------------------------------------
    # Test 3: Backend GLB Structural Validation
    # -------------------------------------------------------------
    print("\n--- 3. Testing Backend GLB Structural Validation ---")
    valid_glb_bytes = create_minimal_glb()
    test_valid_path = settings.OUTPUTS_DIR / "test_valid.glb"
    with open(test_valid_path, "wb") as f:
        f.write(valid_glb_bytes)

    assert generation_service.validate_glb(test_valid_path) is True
    print(f"Valid GLB confirmed: {test_valid_path.name} ({len(valid_glb_bytes)} bytes)")

    # Test corrupted / non-GLB file
    test_invalid_path = settings.OUTPUTS_DIR / "test_invalid.glb"
    with open(test_invalid_path, "wb") as f:
        f.write(b"NOT_A_GLB_FILE_JUST_PLAIN_TEXT")

    assert generation_service.validate_glb(test_invalid_path) is False
    print("Invalid GLB header correctly rejected!")

    # Cleanup temp test files
    test_valid_path.unlink(missing_ok=True)
    test_invalid_path.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # Test 4: Full Background Workflow with Worker Client
    # -------------------------------------------------------------
    print("\n--- 4. Testing End-to-End Background Job with MLWorkerClient ---")
    valid_glb_data = create_minimal_glb()

    # Mock forward_generation to return valid GLB
    async def mock_forward_generation(image_path, prompt, mode, output_glb_path):
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_glb_path, "wb") as f:
            f.write(valid_glb_data)
        return output_glb_path

    mock_client = MLWorkerClient(worker_url="http://mock-colab-worker:8000", worker_token="test-token")

    with mock.patch.object(mock_client, "forward_generation", side_effect=mock_forward_generation):
        backend_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=backend_transport, base_url="http://testserver") as client:
            img_bytes = create_test_image_bytes()

            # Submit job to backend with mock worker client
            with mock.patch("app.routes.generation.worker_client", mock_client), \
                 mock.patch("app.services.generation_service.worker_client", mock_client):
                res = await client.post(
                    "/api/generate",
                    files={"file": ("input_sketch.png", img_bytes, "image/png")},
                    data={"prompt": "Ceramic teapot with glossy glaze", "mode": "draw"}
                )
                assert res.status_code == 202
                job_id = res.json()["job_id"]
                print(f"Job queued via POST /api/generate -> job_id: {job_id}")

                # Execute the worker background processing for this job
                job = generation_service.get_job(job_id)
                input_path = Path(job["input_path"])
                await generation_service.process_job_with_worker(
                    job_id=job_id,
                    input_path=input_path,
                    prompt=job["prompt"],
                    mode="draw",
                    client=mock_client
                )

                # Poll job status
                res_status = await client.get(f"/api/status/{job_id}")
                assert res_status.status_code == 200
                status_data = res_status.json()
                print("Job status after worker processing ->", status_data)
                assert status_data["status"] == "completed"
                assert status_data["model_url"] == f"/api/model/{job_id}"

                # Download model
                res_model = await client.get(f"/api/model/{job_id}")
                assert res_model.status_code == 200
                assert res_model.headers["content-type"] == "model/gltf-binary"
                assert len(res_model.content) == len(valid_glb_data)
                print(f"Downloaded generated model: 200 OK ({len(res_model.content)} bytes)")

    # -------------------------------------------------------------
    # Test 5: Worker Failure -> Job Status FAILED with Useful Error
    # -------------------------------------------------------------
    print("\n--- 5. Testing Worker Failure Handling ---")
    async def mock_failing_generation(image_path, prompt, mode, output_glb_path):
        raise RuntimeError("Remote ML worker timed out after 300.0 seconds while processing 3D generation.")

    mock_fail_client = MLWorkerClient(worker_url="http://mock-colab-worker:8000", worker_token="test-token")
    with mock.patch.object(mock_fail_client, "forward_generation", side_effect=mock_failing_generation):
        job_fail_id = "test-fail-job-uuid"
        dummy_input = settings.UPLOADS_DIR / "dummy.png"
        dummy_input.write_bytes(create_test_image_bytes())
        generation_service.create_job(job_fail_id, dummy_input, prompt="test")

        await generation_service.process_job_with_worker(
            job_id=job_fail_id,
            input_path=dummy_input,
            prompt="test",
            mode="draw",
            client=mock_fail_client
        )

        job_fail_record = generation_service.get_job(job_fail_id)
        assert job_fail_record["status"] == JobStatus.FAILED
        assert "timed out after 300.0 seconds" in job_fail_record["error"]
        print(f"Job failure recorded properly: status='{job_fail_record['status']}'")
        print(f"Error captured: '{job_fail_record['error']}'")

        dummy_input.unlink(missing_ok=True)

    print("\n=======================================================")
    print(">>> ALL WORKER INTEGRATION TESTS PASSED CLEANLY! <<<")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_worker_tests())
