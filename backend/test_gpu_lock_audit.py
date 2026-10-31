"""
Targeted code audit tests for Sketch2Shape AI Colab Worker:
1. GPULock release verification in finally blocks across:
   - Successful inference
   - Pre-thread input validation error
   - Exception raised during inference
   - Async caller timeout
2. asyncio.to_thread() timeout behavior:
   - When asyncio.wait_for() times out, the GPU remains locked (is_busy=True)
     as long as the underlying worker thread is still executing.
   - Simultaneous request receives HTTP 503 while thread is running.
   - Lock is released only when the thread terminates.
   - FastAPI event loop remains unblocked throughout.
"""
import asyncio
import io
import os
from pathlib import Path
import sys
import threading
import time
import unittest.mock as mock

import httpx
from PIL import Image
import trimesh

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent / "techzypher-sketch2shape-ai"))

os.environ["ML_WORKER_TOKEN"] = "audit-secret-token-1234567890"
os.environ["ALLOW_INSECURE_DEV"] = "false"

from worker import app as worker_app, gpu_lock


def create_minimal_glb() -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene(mesh)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def create_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    img.save(buf, format="PNG")
    return buf.getvalue()


async def audit_gpu_lock():
    print("\n=======================================================")
    print("STARTING AUDIT: GPULOCK SAFETY & TIMEOUT CONCURRENCY")
    print("=======================================================\n")

    auth_headers = {"Authorization": f"Bearer {os.environ['ML_WORKER_TOKEN']}"}
    transport = httpx.ASGITransport(app=worker_app)
    img_bytes = create_test_image_bytes()
    valid_glb = create_minimal_glb()

    async with httpx.AsyncClient(transport=transport, base_url="http://worker-test") as client:

        # -------------------------------------------------------------
        # 1. Lock release after SUCCESSFUL inference
        # -------------------------------------------------------------
        print("--- Audit 1: Lock release after successful inference ---")
        assert not gpu_lock.is_locked, "Lock should initially be free"

        def mock_success(sketch_path, prompt, output_dir):
            out_file = Path(output_dir) / "model.glb"
            out_file.write_bytes(valid_glb)
            return str(out_file)

        with mock.patch("worker.generate_3d", side_effect=mock_success):
            res = await client.post(
                "/generate",
                headers=auth_headers,
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            )
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            assert res.headers["content-type"] == "model/gltf-binary"

        # Verify lock was released in finally block
        assert not gpu_lock.is_locked, "Lock was NOT released after successful inference!"
        print("PASSED: Lock released cleanly after successful inference.")

        # -------------------------------------------------------------
        # 2. Lock release after PRE-THREAD validation error
        # -------------------------------------------------------------
        print("\n--- Audit 2: Lock release after pre-thread validation failure ---")
        res_bad = await client.post(
            "/generate",
            headers=auth_headers,
            files={"file": ("empty.png", b"", "image/png")},
            data={"mode": "draw"}
        )
        assert res_bad.status_code == 400
        assert not gpu_lock.is_locked, "Lock was NOT released after validation error!"
        print("PASSED: Lock released cleanly after pre-thread validation error.")

        # -------------------------------------------------------------
        # 3. Lock release after INFERENCE EXCEPTION
        # -------------------------------------------------------------
        print("\n--- Audit 3: Lock release after inference thread exception ---")
        def mock_failing_inference(sketch_path, prompt, output_dir):
            raise RuntimeError("Simulated CUDA out of memory / pipeline error")

        with mock.patch("worker.generate_3d", side_effect=mock_failing_inference):
            res_fail = await client.post(
                "/generate",
                headers=auth_headers,
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            )
            assert res_fail.status_code == 500
            assert "Simulated CUDA" in res_fail.json()["detail"]

        assert not gpu_lock.is_locked, "Lock was NOT released after inference exception!"
        print("PASSED: Lock released cleanly after thread exception.")

        # -------------------------------------------------------------
        # 4. CRITICAL AUDIT: asyncio.wait_for() TIMEOUT & THREAD ISOLATION
        # -------------------------------------------------------------
        print("\n--- Audit 4: Timeout handling and thread persistence ---")
        thread_started = threading.Event()
        can_thread_finish = threading.Event()

        def slow_inference_thread(sketch_path, prompt, output_dir):
            # Signal that the worker thread has started execution on the GPU
            thread_started.set()

            # Block the thread until we allow it to finish
            while not can_thread_finish.is_set():
                time.sleep(0.05)

            out_file = Path(output_dir) / "model.glb"
            out_file.write_bytes(valid_glb)
            return str(out_file)

        # Set a short 0.3s timeout for the test
        with mock.patch("worker.WORKER_TIMEOUT_SECONDS", 0.3), \
             mock.patch("worker.generate_3d", side_effect=slow_inference_thread):

            # Start request 1
            task1 = asyncio.create_task(client.post(
                "/generate",
                headers=auth_headers,
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            ))

            # Wait until thread has started
            while not thread_started.is_set():
                await asyncio.sleep(0.01)

            # Wait for request 1 to timeout
            res1 = await task1
            assert res1.status_code == 504, f"Expected 504 Gateway Timeout, got {res1.status_code}: {res1.text}"
            print("Request 1 timed out with HTTP 504 as expected ->", res1.json()["detail"])

            # THE KEY VERIFICATION:
            # Request 1 has timed out and returned to the caller,
            # BUT slow_inference_thread is STILL RUNNING!
            print("Verifying GPU lock state while background thread is STILL running...")
            assert gpu_lock.is_locked is True, (
                "FATAL: GPU lock was released while background thread was still running on GPU!"
            )
            print("CONFIRMED: gpu_lock.is_locked is True while background thread executes.")

            # Check /health endpoint
            res_health = await client.get("/health")
            assert res_health.status_code == 200
            assert res_health.json()["is_busy"] is True, "Health check should report is_busy=True!"
            print("CONFIRMED: /health reports is_busy: True during background execution.")

            # Now send Request 2 while Request 1's thread is still running
            print("Attempting simultaneous Request 2 while background thread is still running...")
            res2 = await client.post(
                "/generate",
                headers=auth_headers,
                files={"file": ("sketch.png", img_bytes, "image/png")},
                data={"mode": "draw"}
            )
            assert res2.status_code == 503, (
                f"FATAL: Request 2 was NOT rejected! Expected 503, got {res2.status_code}: {res2.text}"
            )
            print("CONFIRMED: Request 2 immediately rejected with HTTP 503 Service Unavailable ->", res2.json()["detail"])

            # Now allow the background thread to finish
            print("Allowing background thread to finish...")
            can_thread_finish.set()
            # Brief pause for threadpool thread to exit finally block
            await asyncio.sleep(0.15)

            # Now the thread has finished: verify lock is released
            assert not gpu_lock.is_locked, "Lock should be released after background thread finishes!"
            print("CONFIRMED: GPU lock released after background thread completed execution.")

            # Check /health again
            res_health2 = await client.get("/health")
            assert res_health2.json()["is_busy"] is False
            print("CONFIRMED: /health reports is_busy: False once thread completes.")

            # -------------------------------------------------------------
            # 5. Subsequent request succeeds after background thread release
            # -------------------------------------------------------------
            print("\n--- Audit 5: Subsequent request succeeds after background thread release ---")
            with mock.patch("worker.WORKER_TIMEOUT_SECONDS", 300.0), \
                 mock.patch("worker.generate_3d", side_effect=mock_success):
                res3 = await client.post(
                    "/generate",
                    headers=auth_headers,
                    files={"file": ("sketch.png", img_bytes, "image/png")},
                    data={"mode": "draw"}
                )
                assert res3.status_code == 200
                print("CONFIRMED: Subsequent request succeeded: HTTP 200 OK.")

    print("\n=======================================================")
    print(">>> ALL AUDIT TESTS PASSED WITH 100% SUCCESS! <<<")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(audit_gpu_lock())
