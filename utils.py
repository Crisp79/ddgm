"""
utils.py
========
Utility functions for the DEM+DGM training pipeline:
  - CelebA dataset loader (subset)
  - FID computation (Frechet Inception Distance)
  - Metric logging / CSV export
  - Sample grid saving
  - Training plot generation
"""

import os
import csv
import math
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def get_celeba_loader(
    root: str,
    subset_size: int = 20_000,
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 4,
    split: str = "train",
) -> DataLoader:
    """
    CelebA dataloader with optional subset.

    Args:
        root        : path to CelebA dataset root (parent of 'celeba/' folder)
        subset_size : number of images to use (≤ full split size). Pass -1 for full.
        batch_size  : mini-batch size
        image_size  : spatial resolution (64 for paper experiments)
        num_workers : DataLoader workers (keep ≤ 4 on laptop)
        split       : 'train' | 'valid' | 'test' | 'all'
    Returns:
        DataLoader
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])

    dataset = datasets.CelebA(root=root, split=split, transform=tf, download=False)

    if 0 < subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    print(f"[Dataset] Using {len(dataset):,} images | {len(loader)} batches/epoch")
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# FID (lightweight, InceptionV3-based)
# ──────────────────────────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """Wraps InceptionV3 to extract pool3 features (2048-d) for FID."""

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        # Remove final classification head; keep up to avgpool
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2), model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2), model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d,
            model.Mixed_6e, model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [-1,1]; resize to 299x299 for Inception
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                             align_corners=False)
        x = (x + 1.0) / 2.0  # -> [0,1]
        return self.features(x).squeeze(-1).squeeze(-1)  # (B, 2048)


def _compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Numerically stable square root of a positive semi-definite matrix."""
    sqrt_A, _ = sqrtm(A, disp=False)
    if np.iscomplexobj(sqrt_A):
        sqrt_A = sqrt_A.real
    return sqrt_A


def compute_fid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
) -> float:
    """
    Frechet Inception Distance between two sets of Inception features.
    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r*Sigma_f))
    """
    mu_r, sig_r = _compute_stats(real_feats)
    mu_f, sig_f = _compute_stats(fake_feats)
    diff = mu_r - mu_f
    cov_mean = _matrix_sqrt(sig_r @ sig_f)
    fid = float(diff @ diff + np.trace(sig_r + sig_f - 2.0 * cov_mean))
    return fid


@torch.no_grad()
def collect_inception_features(
    inception: InceptionFeatureExtractor,
    data_iter,
    n_samples: int,
    device: torch.device,
    is_real: bool = True,
) -> np.ndarray:
    """
    Collect Inception pool3 features from either real images or a generator.

    Args:
        inception  : InceptionFeatureExtractor
        data_iter  : iterable of (images, *) batches  [used if is_real]
        n_samples  : how many samples to collect
        device     : torch device
        is_real    : if True, reads from data_iter; if False, data_iter should be
                     a callable (generator.generate) returning images
    Returns:
        feats: (n_samples, 2048) numpy array
    """
    feats = []
    collected = 0
    for batch in data_iter:
        if collected >= n_samples:
            break
        if is_real:
            imgs = batch[0].to(device)
        else:
            imgs = batch.to(device)
        f = inception(imgs).cpu().numpy()
        feats.append(f)
        collected += f.shape[0]
    return np.concatenate(feats, axis=0)[:n_samples]


# ──────────────────────────────────────────────────────────────────────────────
# Metric CSV logger
# ──────────────────────────────────────────────────────────────────────────────

class MetricLogger:
    """Logs per-epoch metrics to a CSV file and keeps them in memory."""

    FIELDS = ["epoch", "e_loss", "g_loss", "e_real", "e_fake", "gap", "fid", "elapsed_s"]

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []
        # Write header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()

    def log(self, epoch: int, metrics: Dict, elapsed_s: float):
        row = {"epoch": epoch, "elapsed_s": f"{elapsed_s:.1f}"}
        row.update({k: f"{v:.6f}" if isinstance(v, float) else v
                    for k, v in metrics.items()})
        self.records.append(row)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)

    def print_last(self):
        if not self.records:
            return
        r = self.records[-1]
        print(
            f"[Epoch {r['epoch']:>4}] "
            f"E_loss={r['e_loss']:>10}  G_loss={r['g_loss']:>10}  "
            f"E_real={r['e_real']:>10}  E_fake={r['e_fake']:>10}  "
            f"Gap={r['gap']:>10}  FID={r['fid']:>9}  "
            f"({r['elapsed_s']}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sample saving
# ──────────────────────────────────────────────────────────────────────────────

def save_samples(
    generator,
    n_samples: int,
    device: torch.device,
    output_dir: str,
    epoch: int,
    nrow: int = 8,
):
    """Generate and save a grid of images to outputs/samples/."""
    out_dir = Path(output_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        imgs = generator.generate(n_samples, device)   # (N, C, H, W) in [-1,1]
    generator.train()

    grid_path = out_dir / f"epoch_{epoch:04d}.png"
    save_image(imgs, grid_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"  [Samples] Saved {n_samples} images -> {grid_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_metrics(logger: MetricLogger, output_dir: str):
    """
    Plot all tracked metrics after training completes.
    Saves a multi-panel figure to outputs/training_metrics.png.
    """
    if not logger.records:
        print("[Plot] No records to plot.")
        return

    records = logger.records
    epochs  = [int(r["epoch"])         for r in records]
    e_loss  = [float(r["e_loss"])      for r in records]
    g_loss  = [float(r["g_loss"])      for r in records]
    e_real  = [float(r["e_real"])      for r in records]
    e_fake  = [float(r["e_fake"])      for r in records]
    gap     = [float(r["gap"])         for r in records]
    fid     = [float(r["fid"])         for r in records if r["fid"] not in ("nan", "")]

    fid_epochs = [int(r["epoch"]) for r in records if r["fid"] not in ("nan", "")]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DEM + DGM Training Metrics", fontsize=16, fontweight="bold")

    _plot(axes[0, 0], epochs, e_loss,  "Energy Model Loss",      "E_loss",  "steelblue")
    _plot(axes[0, 1], epochs, g_loss,  "Generator Loss",         "G_loss",  "darkorange")
    _plot(axes[1, 0], epochs, e_real,  "Mean Energy (Real)",     "E_real",  "seagreen")
    _plot(axes[1, 1], epochs, e_fake,  "Mean Energy (Fake)",     "E_fake",  "tomato")
    _plot(axes[2, 0], epochs, gap,     "Energy Gap (Real-Fake)", "Gap",     "mediumpurple")

    ax_fid = axes[2, 1]
    if fid:
        ax_fid.plot(fid_epochs, fid, color="goldenrod", linewidth=2, marker="o",
                    markersize=4)
        ax_fid.set_title("Fréchet Inception Distance (FID)", fontweight="bold")
        ax_fid.set_xlabel("Epoch")
        ax_fid.set_ylabel("FID")
        ax_fid.grid(True, alpha=0.3)
    else:
        ax_fid.text(0.5, 0.5, "FID not computed", ha="center", va="center",
                    transform=ax_fid.transAxes, fontsize=12)
        ax_fid.set_title("FID")

    plt.tight_layout()
    out_path = Path(output_dir) / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training metrics saved -> {out_path}")


def _plot(ax, x, y, title, ylabel, color):
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved -> {path}")


def load_checkpoint(path: str, dem, dgm, opt_e, opt_g, device) -> int:
    """Load checkpoint. Returns start epoch."""
    ckpt = torch.load(path, map_location=device)
    dem.load_state_dict(ckpt["dem"])
    dgm.load_state_dict(ckpt["dgm"])
    opt_e.load_state_dict(ckpt["opt_e"])
    opt_g.load_state_dict(ckpt["opt_g"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Resumed from epoch {epoch} -> {path}")
    return epoch