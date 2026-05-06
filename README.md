# Conditional Image Synthesis Toolbox

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified GAN-based framework for conditional image synthesis supporting:

- 🗺️ **Segmentation-conditioned generation** — SPADE normalization at every decoder level  
- 📝 **Text (CLIP) conditioning** — Cross-attention injection from CLIP ViT-B/16/32/L-14 token embeddings  
- 🏗️ **Multi-Scale PatchGAN Discriminator** — 3 discriminators at different resolutions  
- 🎨 **Feature matching + VGG perceptual loss** — Pix2PixHD-style training stability  
- 🔀 **Text interpolation** — Smooth semantic morphing between two text prompts  
- 🎲 **Style variation** — Multiple diverse outputs from the same seg map + text via noise z  
- 📊 **Full training pipeline** — EMA, spectral norm, hinge/lsgan/vanilla GAN loss, W&B/TensorBoard

---

## Architecture Overview

```
Text Prompt ──► CLIP Encoder ──► Token Embeddings (B, 77, 512)
                                        │
Seg Map ──────────────────────────────┐ │ Cross-Attention
(one-hot)                             │ │ at levels 2,3
   │                                  ▼ ▼
   ├──► Encoder ──► Skip 1 ──► Decoder 4 ──► output
   │              ──► Skip 2 ──► Decoder 3 ──►  │   SPADE at
   │              ──► Skip 3 ──► Decoder 2 ──►  │   every level
   │              ──► Skip 4 ──► Decoder 1 ──►  │
   └──────────────────────────────────────────► Tanh ──► [-1,1]
   
   SPADE(feature_map, seg_map) at every decoder block:
     γ(seg), β(seg) = learned spatial scale/bias
     output = γ · Norm(x) + β
```

### Why this design?
- **SPADE** lets the segmentation layout control every spatial location independently — critical for photorealistic scene synthesis
- **CLIP cross-attention** lets text conditions propagate to matching spatial regions (e.g., "rainy" affects sky and road regions differently)
- **Multi-scale PatchGAN** catches both global composition issues and local texture artifacts

---

## Project Structure

```
cond-synthesis/
├── models/
│   ├── generator.py      # UNet generator: SPADE ResBlocks + CLIP cross-attention
│   ├── discriminator.py  # Multi-scale PatchGAN + GAN/feature matching/VGG losses
│   └── clip_encoder.py   # CLIP wrapper (open_clip / transformers / dummy fallback)
├── data/
│   └── datasets.py       # Cityscapes, ADE20K, custom paired datasets
├── utils/
│   └── training.py       # EMA, checkpointing, logging, FID, seg colorization
├── configs/
│   ├── cityscapes.yaml   # Cityscapes street scene synthesis
│   ├── ade20k.yaml       # ADE20K indoor/outdoor synthesis
│   └── debug.yaml        # Tiny config for smoke testing
├── notebooks/
│   └── explore.ipynb     # Text conditioning, interpolation, style variation
├── train.py              # Main training script
└── sample.py             # Inference: single/batch/interpolation
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/cond-synthesis
cd cond-synthesis
pip install -r requirements.txt
```

### 2. Download a dataset

**Cityscapes** (recommended):
```
https://www.cityscapes-dataset.com/
```
Place at `./data/cityscapes/`

**ADE20K**:
```
http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```
Place at `./data/ADEChallengeData2016/`

### 3. Smoke test (no dataset needed)

```bash
# Creates a tiny debug dataset automatically (random noise)
mkdir -p data/debug/images data/debug/segs
python train.py --config configs/debug.yaml
```

### 4. Train

```bash
# Cityscapes
python train.py --config configs/cityscapes.yaml

# ADE20K
python train.py --config configs/ade20k.yaml

# Resume
python train.py --config configs/cityscapes.yaml \
    --resume outputs/cityscapes/ckpts/ckpt_epoch0020_step0050000.pt

# Override config
python train.py --config configs/cityscapes.yaml training.batch_size=2
```

### 5. Generate

```bash
# Single seg map + text prompt
python sample.py \
    --checkpoint outputs/cityscapes/ckpts/best.pt \
    --config configs/cityscapes.yaml \
    --seg_path data/cityscapes/gtFine/val/frankfurt/xxx_gtFine_labelIds.png \
    --text "a rainy night with wet roads"

# Style variations (8 outputs, same text + seg, different noise)
python sample.py \
    --checkpoint outputs/cityscapes/ckpts/best.pt \
    --config configs/cityscapes.yaml \
    --seg_path input/seg.png \
    --text "a sunny summer day" \
    --n_variations 8

# Text interpolation: morph between two prompts
python sample.py \
    --checkpoint outputs/cityscapes/ckpts/best.pt \
    --config configs/cityscapes.yaml \
    --seg_path input/seg.png \
    --text_interp "a bright sunny morning" "a dark stormy night" \
    --n_steps 8

# Batch inference from folder
python sample.py \
    --checkpoint outputs/cityscapes/ckpts/best.pt \
    --config configs/cityscapes.yaml \
    --seg_dir data/cityscapes/gtFine/val/frankfurt/ \
    --text "heavy snowfall"
```

---

## Key Design Choices

| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | SPADE | Spatially-adaptive; preserves semantic layout |
| GAN loss | Hinge | More stable than vanilla BCE; good gradient flow |
| Text injection | Cross-attention | Token-level; each spatial location attends to text |
| Discriminator | Multi-scale PatchGAN | Catches both global and local artifacts |
| Optimizer | Adam (β₁=0, β₂=0.999) | Standard for GAN training; avoids momentum issues |
| Training stability | Spectral norm + feat matching | Prevents mode collapse |

---

## Training Tips

- **Batch size matters**: Use at least 4 for stable GAN training. 8+ is better.  
- **D updates**: 1 D step per G step is standard. Increase to 2 if G loss drops too fast.  
- **CFG analog**: To enable unconditioned generation, occasionally pass null text embeddings (already implemented via `NullTextEmbedding`).  
- **Hinge vs LSGAN**: Hinge is more stable; LSGAN converges faster but can be mode-collapsey.  
- **VGG loss**: Set `lambda_vgg: 0.0` to disable if VGG weights can't be downloaded.

---

## References

- [SPADE (Park et al., 2019)](https://arxiv.org/abs/1903.07291)
- [Pix2PixHD (Wang et al., 2018)](https://arxiv.org/abs/1711.11585)
- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Spectral Normalization (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957)
- [PatchGAN (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
