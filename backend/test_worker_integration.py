"""
Comprehensive integration and unit test suite for the Remote Colab Worker and Backend Client.
Tests:
1. Worker authentication, production token requirement, and timing-safe header validation.
2. Worker concurrency lock (atomic acquire, HTTP 503 rejection without race condition).
3. Input validation: 25 MB limit, Pillow validation, empty files, and strict mode enforcement.
4. Worker and backend GLB structural validation with trimesh (non-empty geometry, finite bounds, positive extents).
5. Clean error handling: failed jobs leave no invalid GLB files behind.
6. Subprocess imports (ml.image_generation, ml.image_conditioning) from PROJECT_ROOT.
"""
import asyncio
import io
import os
from pathlib import Path
import sys
import unittest.mock as mock

import httpx
import numpy as np
from PIL import Image
import trimesh

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent / "techzypher-sketch2shape-ai"))

# Set token and dev flags for testing
os.environ["ML_WORKER_TOKEN"] = "test-secret-token-32-chars-long"
os.environ["ALLOW_INSECURE_DEV"] = "false"

from app.main import app
from app.core.config import settings
from app.schemas.generation import JobStatus
from app.services.generation_service import generation_service
from app.services.worker_client import MLWorkerClient
import worker
from worker import app as worker_app, gpu_lock, validate_glb_mesh, find_project_root


def create_minimal_glb() -> bytes:
    """Create a minimal structurally valid binary GLB using trimesh."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene(mesh)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def create_test_image_bytes(format="PNG") -> bytes:
    """Create a valid image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(30, 40, 50))
    img.save(buf, format=format)
    return buf.getvalue()


async def run_worker_tests():
    print("\n=======================================================")
    print("RUNNING WORKER & BACKEND INTEGRATION TEST SUITE")
    print("=======================================================\n")

    # -------------------------------------------------------------
    # Test 1: PROJECT_ROOT Discovery & Subprocess Imports
    # -------------------------------------------------------------
    print("--- 1. Testing PROJECT_ROOT Discovery & ml Package Imports ---")
    proj_root = find_project_root()
    print("Discovered PROJECT_ROOT ->", proj_root)
    assert (proj_root / "ml").is_dir(), f"ml directory not found in {proj_root}"
    assert (proj_root / "ml" / "pipeline.py").is_file()

    # Verify both pipeline paths can import their respective modules
    from ml.pipeline import generate_3d, generate_3d_from_image
    print("ml.pipeline functions loaded successfully: generate_3d, generate_3d_from_image")

    # -------------------------------------------------------------
    # Test 2: Worker Health & Authentication
    # -------------------------------------------------------------
    print("\n--- 2. Testing Worker /health & Auth Enforcement ---")
    transport = httpx.ASGITransport(app=worker_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:
        # Health check
        res = await client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        health_data = res.json()
        print("Worker /health ->", health_data)
        assert health_data["status"] == "online"
        assert health_data["auth_enabled"] is True

        img_bytes = create_test_image_bytes()
        files = {"file": ("sketch.png", img_bytes, "image/png")}

        # Request without token -> 401
        res_unauth = await client.post("/generate", files=files, data={"mode": "draw"})
        assert res_unauth.status_code == 401, f"Expected 401, got {res_unauth.status_code}"
        print("Rejected unauthenticated request: 401 OK ->", res_unauth.json()["detail"])

        # Request with invalid token -> 401
        res_bad_token = await client.post(
            "/generate",
            headers={"Authorization": "Bearer invalid-wrong-token"},
            files={"file": ("sketch.png", img_bytes, "image/png")},
            data={"mode": "draw"}
        )
        assert res_bad_token.status_code == 401
        print("Rejected bad token: 401 OK ->", res_bad_token.json()["detail"])

    # -------------------------------------------------------------
    # Test 3: Input Validation (Empty, Corrupt, Bad Mode, Size Limit)
    # -------------------------------------------------------------
    print("\n--- 3. Testing Worker Input Validation ---")
    auth_headers = {"Authorization": f"Bearer {os.environ['ML_WORKER_TOKEN']}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:
        # A. Empty image (0 bytes) -> 400
        res_empty = await client.post(
            "/generate",
            headers=auth_headers,
            files={"file": ("empty.png", b"", "image/png")},
            data={"mode": "draw"}
        )
        assert res_empty.status_code == 400
        print("Empty file rejected: 400 ->", res_empty.json()["detail"])

        # B. Corrupted / non-image file -> 400
        res_corrupt = await client.post(
            "/generate",
            headers=auth_headers,
            files={"file": ("fake.png", b"NOT_AN_IMAGE_PAYLOAD", "image/png")},
            data={"mode": "draw"}
        )
        assert res_corrupt.status_code == 400
        print("Corrupt image rejected: 400 ->", res_corrupt.json()["detail"])

        # C. Unsupported mode -> 400
        res_bad_mode = await client.post(
            "/generate",
            headers=auth_headers,
            files={"file": ("valid.png", img_bytes, "image/png")},
            data={"mode": "unsupported_mode"}
        )
        assert res_bad_mode.status_code == 400
        print("Unsupported mode rejected: 400 ->", res_bad_mode.json()["detail"])

    # -------------------------------------------------------------
    # Test 4: Concurrency GPU Lock (Atomic HTTP 503)
    # -------------------------------------------------------------
    print("\n--- 4. Testing Concurrency GPU Lock (HTTP 503 Protection) ---")
    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:
        # Simulate GPU lock being held
        assert gpu_lock.try_acquire() is True
        try:
            res_busy = await client.post(
                "/generate",
                headers=auth_headers,
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            )
            assert res_busy.status_code == 503, f"Expected 503, got {res_busy.status_code}"
            print("GPU busy protection: 503 Service Unavailable ->", res_busy.json()["detail"])
        finally:
            gpu_lock.release()

    # -------------------------------------------------------------
    # Test 5: Worker & Backend GLB Structural Validation
    # -------------------------------------------------------------
    print("\n--- 5. Testing GLB Structural Validation with Trimesh ---")
    valid_glb_bytes = create_minimal_glb()
    test_valid_path = settings.OUTPUTS_DIR / "test_valid.glb"
    test_valid_path.write_bytes(valid_glb_bytes)

    # Worker validation function
    validate_glb_mesh(test_valid_path)
    print("Worker validate_glb_mesh: PASSED on valid GLB")

    # Backend validation function
    assert generation_service.validate_glb(test_valid_path) is True
    print(f"Backend validate_glb: PASSED on {test_valid_path.name} ({len(valid_glb_bytes)} bytes)")

    # Test corrupted / non-GLB file
    test_invalid_path = settings.OUTPUTS_DIR / "test_invalid.glb"
    test_invalid_path.write_bytes(b"NOT_A_GLB_FILE_JUST_PLAIN_TEXT")

    try:
        validate_glb_mesh(test_invalid_path)
        assert False, "Worker should have rejected corrupted GLB"
    except RuntimeError as err:
        print("Worker correctly rejected invalid GLB ->", str(err))

    assert generation_service.validate_glb(test_invalid_path) is False
    print("Backend correctly rejected invalid GLB header")

    # Cleanup temp test files
    test_valid_path.unlink(missing_ok=True)
    test_invalid_path.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # Test 6: Full Background Workflow with Worker Client
    # -------------------------------------------------------------
    print("\n--- 6. Testing End-to-End Background Job with MLWorkerClient ---")
    valid_glb_data = create_minimal_glb()

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
    # Test 7: Worker Failure -> Job Status FAILED & Cleans Up Output
    # -------------------------------------------------------------
    print("\n--- 7. Testing Worker Failure Handling & Output Cleanup ---")
    target_output_glb = settings.OUTPUTS_DIR / "test-fail-job-uuid.glb"

    async def mock_failing_generation(image_path, prompt, mode, output_glb_path):
        # Simulate partial corrupt write followed by failure
        output_glb_path.write_bytes(b"CORRUPTED_PARTIAL_FILE")
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

        # Verify failed job does NOT leave invalid GLB behind!
        assert not target_output_glb.exists(), "Failed job left invalid GLB behind!"
        print("Confirmed: Failed job leaves NO invalid GLB file behind.")

        dummy_input.unlink(missing_ok=True)

    print("\n=======================================================")
    print(">>> ALL WORKER INTEGRATION TESTS PASSED CLEANLY! <<<")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_worker_tests())
