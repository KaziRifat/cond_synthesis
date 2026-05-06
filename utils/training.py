"""
Training utilities:
  - EMA for generator
  - Checkpointing (generator + discriminator)
  - WandB / TensorBoard logging
  - FID score
  - Sample grid saving
  - Gradient utilities
"""

import os
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of generator weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.original = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name])
        self.original.clear()

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, d):
        self.shadow = d['shadow']
        self.decay  = d.get('decay', self.decay)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: str,
    epoch: int,
    step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    ema: EMA,
    best: bool = False,
    config: Optional[Dict] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    fname = 'best.pt' if best else f'ckpt_epoch{epoch:04d}_step{step:07d}.pt'
    path  = os.path.join(output_dir, fname)

    torch.save({
        'epoch': epoch,
        'step':  step,
        'generator_state_dict':     generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'ema_state_dict':   ema.state_dict(),
        'config': config,
    }, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: Optional[torch.optim.Optimizer] = None,
    opt_d: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt['generator_state_dict'])
    discriminator.load_state_dict(ckpt['discriminator_state_dict'])
    if opt_g: opt_g.load_state_dict(ckpt['opt_g_state_dict'])
    if opt_d: opt_d.load_state_dict(ckpt['opt_d_state_dict'])
    if ema:   ema.load_state_dict(ckpt['ema_state_dict'])
    print(f"Loaded: {path}  (epoch {ckpt.get('epoch','?')}, step {ckpt.get('step','?')})")
    return ckpt


# ---------------------------------------------------------------------------
# Sample grid
# ---------------------------------------------------------------------------

def save_sample_grid(
    real: torch.Tensor,
    fake: torch.Tensor,
    seg: torch.Tensor,
    path: str,
    nrow: int = 4,
    num_seg_classes: int = 19,
):
    """Save a 3-column grid: seg colormap | real image | fake image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Colorize segmentation map
    seg_colored = colorize_seg(seg[:nrow], num_seg_classes)  # (N, 3, H, W) in [0,1]

    real_01 = (real[:nrow].clamp(-1, 1) + 1) / 2
    fake_01 = (fake[:nrow].clamp(-1, 1) + 1) / 2

    # Interleave: seg, real, fake for each sample
    cols = []
    for i in range(min(nrow, real.shape[0])):
        cols.extend([seg_colored[i], real_01[i], fake_01[i]])

    grid = make_grid(torch.stack(cols), nrow=3, padding=2)
    save_image(grid, path)
    return path


def colorize_seg(seg: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert (B, num_classes, H, W) one-hot or (B, 1, H, W) label map
    to (B, 3, H, W) RGB using a fixed color palette.
    """
    if seg.shape[1] > 1:
        seg = seg.argmax(dim=1, keepdim=True)  # (B, 1, H, W)

    B, _, H, W = seg.shape
    palette = _make_palette(num_classes)          # (num_classes, 3)
    palette = palette.to(seg.device)

    seg_flat = seg.squeeze(1).long().view(-1)     # (BHW,)
    seg_flat = seg_flat.clamp(0, num_classes - 1)
    colors   = palette[seg_flat]                  # (BHW, 3)
    colored  = colors.view(B, H, W, 3).permute(0, 3, 1, 2).float() / 255.0
    return colored


def _make_palette(num_classes: int) -> torch.Tensor:
    """Generate a deterministic color palette."""
    palette = torch.zeros(num_classes, 3, dtype=torch.uint8)
    for i in range(num_classes):
        r, g, b, idx = 0, 0, 0, i
        for j in range(8):
            r |= ((idx >> 0) & 1) << (7 - j)
            g |= ((idx >> 1) & 1) << (7 - j)
            b |= ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        palette[i] = torch.tensor([r, g, b])
    return palette


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, config: Dict, project: str = 'cond-synthesis',
                 use_wandb: bool = True, use_tb: bool = True):
        self.wandb_run = None
        self.tb_writer = None
        self._step = 0

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project, config=config, resume='allow')
                print(f"WandB: {self.wandb_run.url}")
            except Exception as e:
                print(f"WandB failed ({e})")

        if use_tb and not self.wandb_run:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(config.get('output_dir', 'outputs'), 'tb_logs')
                self.tb_writer = SummaryWriter(tb_dir)
                print(f"TensorBoard: {tb_dir}")
            except Exception as e:
                print(f"TensorBoard unavailable ({e})")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        step = step if step is not None else self._step
        self._step = step + 1
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        print(f"[step {step:7d}] " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        step = step if step is not None else self._step
        if self.wandb_run:
            try:
                import wandb
                grid = make_grid(images, normalize=True, value_range=(0, 1))
                self.wandb_run.log({tag: wandb.Image(grid.permute(1, 2, 0).cpu().numpy())}, step=step)
            except Exception:
                pass
        if self.tb_writer:
            grid = make_grid(images, normalize=True, value_range=(0, 1))
            self.tb_writer.add_image(tag, grid, step)

    def finish(self):
        if self.wandb_run: self.wandb_run.finish()
        if self.tb_writer:  self.tb_writer.close()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_dir: str, fake_dir: str, device: str = 'cuda') -> float:
    try:
        from cleanfid import fid
        score = fid.compute_fid(real_dir, fake_dir, device=device)
        print(f"FID: {score:.2f}")
        return score
    except ImportError:
        print("FID: pip install clean-fid")
        return -1.0


def save_images_for_fid(images: torch.Tensor, out_dir: str, start_idx: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    imgs_01 = (images.clamp(-1, 1) + 1) / 2
    for i, img in enumerate(imgs_01):
        save_image(img, os.path.join(out_dir, f'{start_idx + i:06d}.png'))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(model: nn.Module) -> float:
    total = sum(
        p.grad.detach().norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    )
    return total ** 0.5


class WarmupLinearScheduler:
    def __init__(self, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return max(0.0, 1.0 - progress)
