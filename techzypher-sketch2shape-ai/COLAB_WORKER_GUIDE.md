# Google Colab GPU Worker Deployment Guide

This guide details how to launch the remote GPU worker in Google Colab (T4 GPU runtime) and link your local Sketch2Shape AI FastAPI backend to it.

---

## 🏗️ Architecture

```
[React Frontend] (localhost:5173)
       │  POST /api/generate
       ▼
[FastAPI Backend] (localhost:8000)
       │  POST /generate (multipart/form-data: file + prompt + mode)
       │  Authorization: Bearer <ML_WORKER_TOKEN>
       ▼  (Secure Tunnel: Cloudflare Tunnel or ngrok)
[Google Colab GPU Worker] (FastAPI, T4 GPU)
       │  GPULock (atomic non-blocking, returns HTTP 503 if busy)
       ├── Preprocessing (ml.preprocessing)
       ├── SD 1.5 + ControlNet Scribble (controlnet-env: Python 3.10)
       ├── TripoSR 3D Mesh Reconstruction (triposr-env: Python 3.10)
       └── Mesh Validation & PBR Material Export (ml.mesh_utils)
       │
       ▼  Returns validated binary model.glb
[FastAPI Backend] saves backend/outputs/{job_id}.glb -> marks COMPLETED -> Three.js renders in frontend
```

---

## 🚀 Step 1: Open Google Colab & Select T4 GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. In the top menu, go to **Runtime** → **Change runtime type** → select **T4 GPU** → click **Save**.
3. Verify GPU availability:
   ```python
   import torch
   assert torch.cuda.is_available(), "Please enable GPU in Runtime -> Change runtime type"
   print("GPU:", torch.cuda.get_device_name(0))
   ```

---

## 📂 Step 2: Make Project Files Available & Locate PROJECT_ROOT

The worker requires the `ml/` directory and `worker.py`. In Colab, clone the repository or upload your files:

```bash
# Clone the repository to /content/techzypher
if [ ! -d "/content/techzypher" ]; then
  git clone https://github.com/arpit-thorat07/techzypher.git /content/techzypher
fi
```

Verify that `PROJECT_ROOT` directly contains the `ml/` package and `worker.py`:

```python
from pathlib import Path

# Search for the directory containing ml/ and worker.py
candidates = [
    Path("/content/techzypher"),
    Path("/content/techzypher-sketch2shape-ai"),
    Path("/content"),
]
project_root = None
for c in candidates:
    if (c / "ml").is_dir() and (c / "worker.py").is_file():
        project_root = c
        break

assert project_root is not None, "Could not find project root containing ml/ and worker.py"
print(f"Verified PROJECT_ROOT at: {project_root}")
```

---

## 📦 Step 3: Set Up Environments & Dependencies

Run the following setup in Colab (or run the cells in `notebooks/colab_worker.ipynb`):

```bash
# 1. Install uv fast package manager and Python 3.10
pip install -q uv
uv python install 3.10

# 2. Create isolated virtual environments for ControlNet and TripoSR
uv venv /content/controlnet-env --python 3.10
uv venv /content/triposr-env --python 3.10

# 3. Install PyTorch 2.1.2 with CUDA 12.1 into both isolated environments
uv pip install --python /content/controlnet-env/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /content/triposr-env/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install ControlNet dependencies into controlnet-env
uv pip install --python /content/controlnet-env/bin/python \
    numpy==1.26.4 diffusers==0.27.2 transformers==4.38.2 \
    accelerate==0.29.3 huggingface-hub==0.25.2 safetensors Pillow==10.2.0

# 5. Clone TripoSR & install its dependencies into triposr-env
if [ ! -d "/content/TripoSR" ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR.git /content/TripoSR
fi

python -c "
lines = open('/content/TripoSR/requirements.txt').read().splitlines()
lines = [l for l in lines if 'torchmcubes' not in l and l.strip() != 'gradio' and not l.startswith('huggingface-hub')]
lines += ['huggingface-hub==0.17.3', 'numpy==1.26.4', 'scikit-image==0.22.0', 'onnxruntime-gpu']
open('/content/triposr-req.txt', 'w').write('\n'.join(lines) + '\n')
"
uv pip install --python /content/triposr-env/bin/python -r /content/triposr-req.txt

# 6. Apply TripoSR marching_cubes patch
python -c "
path = '/content/TripoSR/tsr/models/isosurface.py'
src = open(path).read()
old = 'from torchmcubes import marching_cubes'
rep = '''from skimage.measure import marching_cubes as _skimage_mc
import numpy as np
import torch

def marching_cubes(volume, level):
    verts, faces, _, _ = _skimage_mc(volume.detach().float().cpu().numpy(), level=level)
    verts = np.ascontiguousarray(verts[:, ::-1])
    faces = np.ascontiguousarray(faces.astype(np.int64))
    return torch.from_numpy(verts), torch.from_numpy(faces)
'''
if old in src:
    open(path, 'w').write(src.replace(old, rep))
    print('Marching cubes patch applied!')
elif '_skimage_mc' in src:
    print('Marching cubes already patched.')
"

# 7. Install Worker server dependencies in main Colab environment
pip install -q fastapi uvicorn python-multipart trimesh Pillow
```

---

## 🔒 Step 4: Configure Secure Worker Authentication

Authentication is required for the remote worker. Never use a hardcoded token or print the plain token to logs.

You can configure the token via **Colab Secrets** (recommended) or generate a strong random token programmatically:

```python
import os
import secrets

# Option A: Retrieve token from Colab Secrets (Add 'ML_WORKER_TOKEN' in the Secrets tab on the left)
try:
    from google.colab import userdata
    token = userdata.get("ML_WORKER_TOKEN")
except Exception:
    token = None

# Option B: Generate a cryptographically strong token if not in Colab Secrets
if not token:
    token = secrets.token_urlsafe(32)

# Export to environment (do NOT print or log the raw token value)
os.environ["ML_WORKER_TOKEN"] = token
os.environ["PROJECT_ROOT"] = str(project_root)
os.environ["CONTROLNET_PYTHON"] = "/content/controlnet-env/bin/python"
os.environ["TRIPOSR_PYTHON"] = "/content/triposr-env/bin/python"
os.environ["TRIPOSR_ROOT"] = "/content/TripoSR"

# Save configuration securely to a local file for copying to your backend
with open("/content/worker_config.env", "w") as f:
    f.write(f"ML_WORKER_TOKEN={token}\n")

print("Worker environment configured. Authentication token generated and stored securely.")
```

---

## 🌐 Step 5: Expose Port 8000 via Cloudflare Tunnel

Download and start `cloudflared` to route traffic to the worker:

```bash
# Download and install cloudflared
wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1
```

Start the tunnel in background and extract the public HTTPS URL:

```python
import subprocess, time, re

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stderr=subprocess.PIPE,
    text=True,
)

public_url = None
for _ in range(30):
    line = tunnel_proc.stderr.readline()
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        break
    time.sleep(0.5)

print("=" * 60)
print("PUBLIC WORKER TUNNEL URL:", public_url)
print("=" * 60)
```

---

## ⚡ Step 6: Start Worker Service

Launch the worker from the directory containing `worker.py`:

```bash
cd "$PROJECT_ROOT"
python worker.py
```

The worker service will start on port 8000. It will:
- Reject startup if `ML_WORKER_TOKEN` is missing.
- Validate all incoming requests with timing-safe comparison.
- Enforce the non-blocking `GPULock` (returning HTTP 503 if another generation is active).
- Validate input file format, dimensions, size limit (25 MB), and mode.
- Validate the generated GLB mesh with `trimesh` (non-empty geometry, finite bounds, positive extents) before returning.

---

## 💻 Step 7: Configure Local FastAPI Backend

In your local project on Windows:

1. Create or edit `D:\Hypertune\backend\.env`:
   ```ini
   ML_WORKER_URL=https://<your-cloudflare-subdomain>.trycloudflare.com
   ML_WORKER_TOKEN=<your-token-from-worker_config.env>
   ML_WORKER_TIMEOUT_SECONDS=300.0
   ```
2. Start the local backend:
   ```powershell
   cd D:\Hypertune\backend
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Test connectivity at `http://127.0.0.1:8000/api/health`. You will see:
   ```json
   {
     "status": "healthy",
     "worker": {
       "configured": true,
       "reachable": true,
       "status": "online",
       "is_busy": false
     }
   }
   ```
4. Draw or upload an image in the React frontend and click **Generate 3D Model**. The job is processed on the Colab T4 GPU and the validated GLB is rendered automatically.
