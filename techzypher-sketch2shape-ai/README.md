# Sketch2Shape AI

Sketch-to-3D prototype developed for TechZephyr 4.0. Supports line sketches or colored images with an optional descriptive text prompt; the ML pipeline produces a GLB asset.

## Pipeline

1. `ml.preprocessing.preprocess_sketch`: crops and centers dark strokes in a 512×512 ControlNet scribble (white on black).
2. `ml.image_generation.generate_image`: Stable Diffusion v1.5 + ControlNet Scribble, deterministic seed 42.
3. `ml.image_conditioning.condition_image`: optional Stable Diffusion img2img conditioning for colored-image + prompt input. This can alter the input appearance; the prompt does not act directly on TripoSR.
4. `ml.reconstruction.reconstruct_3d`: TripoSR single-image 3D reconstruction.
5. `ml.mesh_utils.validate_and_export`: verifies mesh bounds, repairs negative signed volume where possible, and exports GLB with an explicit white-base PBR material for vertex-colored meshes.
6. `ml.pipeline.generate_3d(sketch_path, prompt, output_dir)`: sketch + prompt workflow.
7. `ml.pipeline.generate_3d_from_image(image_path, prompt, output_dir)`: colored image + prompt workflow.

The code defaults to Google Colab paths (`/content/controlnet-env`, `/content/triposr-env`, `/content/TripoSR`) and supports overrides via environment variables (`CONTROLNET_PYTHON`, `TRIPOSR_PYTHON`, `TRIPOSR_ROOT`, `PROJECT_ROOT`). It is designed to run on a cloud GPU (e.g. Google Colab with a T4 GPU) as a remote worker service for the FastAPI backend.

## Remote Colab GPU Worker

A dedicated FastAPI worker API service is provided in `worker.py` and `notebooks/colab_worker.ipynb`:
- **`GET /health`**: Returns GPU status and worker readiness.
- **`POST /generate`**: Accepts uploaded image, prompt, and mode (`draw` or `upload`), authenticates against `ML_WORKER_TOKEN`, enforces a single-request GPU concurrency lock, runs the ControlNet + TripoSR pipeline, and returns the validated binary GLB model.
- See [COLAB_WORKER_GUIDE.md](COLAB_WORKER_GUIDE.md) for step-by-step instructions.

## Requirements

Use Colab with a T4 GPU and two isolated Python 3.10 environments. The notebook/orchestrator Python also requires `numpy`, `Pillow`, and `trimesh`.

**ControlNet environment:**
- torch 2.1.2 + torchvision 0.16.2 (CUDA 12.1 wheels)
- numpy 1.26.4, Pillow 10.2.0
- diffusers 0.27.2, transformers 4.38.2, accelerate 0.29.3, huggingface-hub 0.25.2, safetensors

**TripoSR environment:**
- Official [TripoSR repository](https://github.com/VAST-AI-Research/TripoSR), Python 3.10
- torch 2.1.2 + torchvision 0.16.2 (CUDA 12.1 wheels)
- TripoSR's requirements excluding `gradio` and `torchmcubes`
- huggingface-hub 0.17.3, numpy 1.26.4, scikit-image 0.22.0, onnxruntime 1.19.2
- Colab CUDA extension build failed for `torchmcubes`; testing used a `skimage.measure.marching_cubes` fallback patched into `tsr/models/isosurface.py` in the TripoSR checkout. This patch is not included in this repository because TripoSR remains an external dependency.

Inference may download pretrained model weights on the first run. Model files are not committed.

## Usage (after setting up both environments)

From the Colab notebook, with the repository checked out at `/content` or otherwise added to Python's import path:

```python
from ml.pipeline import generate_3d

path = generate_3d(
    "/content/chair.png",
    "a single wooden chair, realistic product photo, full object visible",
    "/content/outputs",
)
print(path)
```

For a colored input image with prompt guidance:

```python
from ml.pipeline import generate_3d_from_image

path = generate_3d_from_image(
    "/content/controller.png",
    "a pink gaming controller with purple buttons, rounded ergonomic body, studio lighting",
    "/content/outputs",
)
print(path)
```

The returned path points to a job-specific `model.glb`. The sketch workflow saves `control.png` and `generated.png`; the colored-image workflow saves `conditioned.png`.

## Test observations and limitations

A chair, stool, mug, and desk lamp were used during manual tests. GLB creation was successful, but successful mesh export is **not** proof of correct 3D reconstruction. Notably, the mug's hollow interior and lamp shade depth were not consistently preserved. TripoSR infers unseen geometry from one image, and prompts alone may not solve cavities and thin parts. Review each model in a 3D viewer. The exported GLB carries vertex colors (COLOR_0) with a supporting material, not a UV-baked texture image; some viewers may require material-preview mode. Prompt-guided img2img can change colors or proportions, and does not guarantee preservation of hidden geometry.

## Model attribution

This project uses externally pretrained work, not models trained from scratch:
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet Scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble)
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)

Check each upstream model's licensing and terms before redistribution or deployment. Do not commit model weights or private tokens.
