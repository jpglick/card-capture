# Vast.ai Base Template Setup

## Automated setup (recommended)

Uses **GitHub Container Registry** (`ghcr.io`) — no Docker Hub account needed, just your existing GitHub credentials.

**Prerequisites:** GitHub PAT (classic) with `write:packages`, `read:packages`, `delete:packages` scopes.
Create at: **github.com → Settings → Developer settings → Personal access tokens (classic)**

```bash
# One-time setup — builds Docker image, pushes to ghcr.io, updates config
python3 scripts/setup_cuda_template.py --docker-user YOUR_GITHUB_USERNAME

# With smoke test (spins up a real instance, tests health endpoint, destroys it)
VAST_API_KEY=your_key python3 scripts/setup_cuda_template.py --docker-user YOUR_GITHUB_USERNAME --verify

# Re-run any time — safe to run again (skips push if image unchanged)
python3 scripts/setup_cuda_template.py --docker-user YOUR_GITHUB_USERNAME
```

The script prompts for your GitHub PAT on first run, logs in to `ghcr.io`, and writes `vast_template_id` into `card_capture_config.json` automatically.

---

## Manual fallback — what to install on the base instance before saving the template

1. Start a vast.ai instance with a PyTorch+CUDA base image:
   `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`

2. Clone the repo:
   ```
   git clone https://github.com/jpglick/card-capture.git /workspace/card-capture
   ```

3. Install heavy dependencies (these are baked into the template):
   ```
   cd /workspace/card-capture
   pip install -e '.[model]'
   pip install decord ultralytics kornia
   ```

4. Download the YOLO model so it's cached:
   ```
   python -c "from card_capture.detectors import CardcaptorUltralyticsDetector; CardcaptorUltralyticsDetector(device='cuda')._load_model()"
   ```

5. Download DINOv2 so it's cached:
   ```
   python -c "from card_capture.ml.models.dino_embedder import DinoEmbedder; DinoEmbedder()"
   ```

6. Save the instance as a template in the vast.ai console.
   Copy the template ID into `card_capture_config.json` → `"vast_template_id"`.

## On each boot (handled automatically by vastai_worker startup script)

```bash
cd /workspace/card-capture && git pull origin main -q
pip install -e '.[app]' -q
uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765
```
