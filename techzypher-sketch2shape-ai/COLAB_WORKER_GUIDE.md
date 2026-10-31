# Google Colab GPU Worker Deployment Guide

This guide explains how to start the remote GPU worker in Google Colab and connect your local FastAPI backend to it.

---

## 🏗️ Architecture

```
[React Frontend] (localhost:5173)
       │  HTTP /api/generate
       ▼
[FastAPI Backend] (localhost:8000)
       │  POST /generate (Forward file + prompt + mode)
       │  Authorization: Bearer <ML_WORKER_TOKEN>
       ▼  (Secure Tunnel: Cloudflare or ngrok)
[Google Colab GPU Worker] (T4 GPU, 15GB VRAM)
       │
       ├── Preprocessing (ml.preprocessing)
       ├── SD 1.5 + ControlNet Scribble (controlnet-env)
       ├── TripoSR 3D Mesh Reconstruction (triposr-env)
       └── Mesh Validation & PBR Material Export (ml.mesh_utils)
       │
       ▼  Returns validated binary model.glb
[FastAPI Backend] saves backend/outputs/{job_id}.glb -> marks COMPLETED -> serves to frontend
```

---

## 🚀 Step 1: Open Google Colab & Select T4 GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload the `techzypher-sketch2shape-ai` project or clone the repository to `/content`.
3. In the menu, go to **Runtime** → **Change runtime type** → select **T4 GPU** → click **Save**.

---

## 📦 Step 2: Set Up Environments & Worker Dependencies

In Colab, run the following commands (or run the cells in `notebooks/colab_worker.ipynb`):

```bash
# 1. Install uv package manager and Python 3.10
pip install -q uv
uv python install 3.10

# 2. Create isolated virtual environments
uv venv /content/controlnet-env --python 3.10
uv venv /content/triposr-env --python 3.10

# 3. Install PyTorch 2.1.2 with CUDA 12.1 in both environments
uv pip install --python /content/controlnet-env/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /content/triposr-env/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install ControlNet dependencies
uv pip install --python /content/controlnet-env/bin/python \
    numpy==1.26.4 diffusers==0.27.2 transformers==4.38.2 \
    accelerate==0.29.3 huggingface-hub==0.25.2 safetensors Pillow==10.2.0

# 5. Clone TripoSR & install its dependencies
git clone https://github.com/VAST-AI-Research/TripoSR.git /content/TripoSR
python -c "
lines = open('/content/TripoSR/requirements.txt').read().splitlines()
lines = [l for l in lines if 'torchmcubes' not in l and l.strip() != 'gradio' and not l.startswith('huggingface-hub')]
lines += ['huggingface-hub==0.17.3', 'numpy==1.26.4', 'scikit-image==0.22.0', 'onnxruntime-gpu']
open('/content/triposr-req.txt', 'w').write('\n'.join(lines) + '\n')
"
uv pip install --python /content/triposr-env/bin/python -r /content/triposr-req.txt

# 6. Apply TripoSR CPU marching_cubes patch
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
"

# 7. Install Worker API server dependencies
pip install -q fastapi uvicorn python-multipart trimesh
```

---

## 🔒 Step 3: Configure Worker Token & Launch

Choose a secret token (e.g. `sketch2shape-secret-token-2026`).

```python
import os

# Set environment variables
os.environ["ML_WORKER_TOKEN"] = "sketch2shape-secret-token-2026"
os.environ["CONTROLNET_PYTHON"] = "/content/controlnet-env/bin/python"
os.environ["TRIPOSR_PYTHON"] = "/content/triposr-env/bin/python"
os.environ["TRIPOSR_ROOT"] = "/content/TripoSR"
os.environ["PROJECT_ROOT"] = "/content"
```

### Expose Port 8000 via Cloudflare Tunnel (Free, No signup required)

```bash
# Download cloudflared
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Start tunnel in background
import subprocess, time, re
tunnel_proc = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"], stderr=subprocess.PIPE, text=True)

for _ in range(30):
    line = tunnel_proc.stderr.readline()
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        print("PUBLIC WORKER URL:", match.group(0))
        break
    time.sleep(0.5)
```

### Start the Worker Service

```bash
python /content/worker.py
```

---

## 💻 Step 4: Configure Local FastAPI Backend

In your local project on Windows:

1. Open `D:\Hypertune\backend\.env` (or create it from `backend/.env.example`):
   ```ini
   ML_WORKER_URL=https://<your-subdomain>.trycloudflare.com
   ML_WORKER_TOKEN=sketch2shape-secret-token-2026
   ML_WORKER_TIMEOUT_SECONDS=300.0
   ```
2. Start the local backend:
   ```bash
   cd D:\Hypertune\backend
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Check worker readiness at `http://127.0.0.1:8000/api/health`. You will see:
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
4. Now, when a sketch or image is submitted via the React frontend or Swagger UI (`POST /api/generate`), the backend forwards it to the Colab GPU worker, downloads the real validated GLB, and serves it automatically to the Three.js viewport!
