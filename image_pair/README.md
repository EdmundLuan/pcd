# Constrained Coupled Image Pair Generation

This implementation supports constrained joint image generation with:
- projection to exemplar-based latent convex hulls
- coupling guidance from a cost function
- post-generation evaluation (classifier confidence, LPIPS, CLIP similarity)

Use the scripts in `scripts/` as the primary entrypoints.

---

## Installation

### Requirements
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html) (or equivalent)

### Steps

1. Create and activate an environment:
   ```bash
   conda create -n pcd-im python=3.10 -y
   conda activate pcd-im
   ```
2. Install PyTorch (example for CUDA 12.8; adjust to your machine):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pretrained Weights

Use the setup script to download required checkpoints from Google Drive:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

- This downloads:
  - latent classifier checkpoint for coupling gradients
  - image classifier checkpoint for post-run analysis
- Files are placed directly into the expected `model_weights/.../checkpoints_*/` paths.

To force a fresh download:

```bash
./scripts/setup.sh --force
```

---

## Inference

Run:

```bash
chmod +x scripts/main.sh
./scripts/main.sh
```

Notes:
- Main runtime configuration is in `scripts/main.sh`.
- This includes model IDs, exemplar paths, projection/coupling switches, and metric classifier paths.
- Outputs are written under:
  - `outputs/<setup_name>/<model_name>/`

---

## Result Analysis

Analysis is integrated into the main run (`src/main.py`):
- classifier confidence JSON
- intra-batch LPIPS JSON
- LPIPS-to-exemplar JSON
- CLIP similarity JSON

All metrics are saved into the same run output directory.

---

## ✏️ Citation

If this codebase is useful for your research, please consider citing:

```bibtex
@inproceedings{luan2026projected,
    title={Projected Coupled Diffusion for Test-Time Constrained Joint Generation},
    author={Hao Luan and Yi Xian Goh and See-Kiong Ng and Chun Kai Ling},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=1FEm5JLpvg}
}
```

---

## References

- Luan H, Goh YX, Ng SK, Ling CK. Projected Coupled Diffusion for Test-Time Constrained Joint Generation. ICLR 2026.
