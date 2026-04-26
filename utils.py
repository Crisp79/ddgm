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
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_scalar(value: str):
    """Parse a scalar from a simple YAML-like key:value value string."""
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_base_config(config_path: Path) -> Dict:
    """Load a flat key:value config file such as config/base.yaml."""
    cfg = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split(":", 1)
            cfg[key.strip()] = parse_scalar(val)
    return cfg


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


def compute_precision_recall(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    k: int = 10,
) -> tuple[float, float]:
    """
    Compute improved precision and recall on feature embeddings.

    Precision: fraction of generated samples that lie within the manifold of real data.
    Recall: fraction of real samples covered by the generated manifold.
    """
    if len(real_feats) <= k or len(fake_feats) <= k:
        return float("nan"), float("nan")

    real_feats = np.asarray(real_feats)
    fake_feats = np.asarray(fake_feats)

    rr = _pairwise_distances(real_feats, real_feats)
    np.fill_diagonal(rr, np.inf)
    r_radius = np.partition(rr, k - 1, axis=1)[:, k - 1]

    ff = _pairwise_distances(fake_feats, fake_feats)
    np.fill_diagonal(ff, np.inf)
    f_radius = np.partition(ff, k - 1, axis=1)[:, k - 1]

    rf = _pairwise_distances(real_feats, fake_feats)

    precision = np.mean(np.any(rf <= r_radius[:, None], axis=0))
    recall = np.mean(np.any(rf <= f_radius[None, :], axis=1))
    return float(precision), float(recall)


def _pairwise_distances(x: np.ndarray, y: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Compute pairwise Euclidean distances without external dependencies."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T

    distances = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        d2 = x_norm[start:end] + y_norm - 2.0 * (x[start:end] @ y.T)
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        distances[start:end] = d2
    return distances


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

    FIELDS = [
        "epoch",
        "e_loss",
        "g_loss",
        "e_real",
        "e_fake",
        "e_real_var",
        "e_fake_var",
        "gap",
        "fid",
        "precision",
        "recall",
        "elapsed_s",
    ]

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
    epochs      = [int(r["epoch"]) for r in records]
    e_loss      = [float(r["e_loss"]) for r in records]
    g_loss      = [float(r["g_loss"]) for r in records]
    e_real      = [float(r["e_real"]) for r in records]
    e_fake      = [float(r["e_fake"]) for r in records]
    e_real_var  = [float(r.get("e_real_var", "nan")) for r in records]
    e_fake_var  = [float(r.get("e_fake_var", "nan")) for r in records]
    gap         = [float(r["gap"]) for r in records]
    fid         = [float(r["fid"]) for r in records if r["fid"] not in ("nan", "")]
    precision   = [float(r["precision"]) for r in records if r.get("precision", "nan") not in ("nan", "")]
    recall      = [float(r["recall"]) for r in records if r.get("recall", "nan") not in ("nan", "")]

    fid_epochs = [int(r["epoch"]) for r in records if r["fid"] not in ("nan", "")]
    pr_epochs = [int(r["epoch"]) for r in records if r.get("precision", "nan") not in ("nan", "")]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("DEM + DGM Training Metrics", fontsize=16, fontweight="bold")

    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, e_loss, color="steelblue", linewidth=2, label="E_loss")
    ax_loss.plot(epochs, g_loss, color="darkorange", linewidth=2, label="G_loss")
    ax_loss.set_title("Energy / Generator Loss", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    _plot(axes[0, 1], epochs, e_real,      "Mean Energy (Real)",   "E_real",     "seagreen")
    _plot(axes[0, 2], epochs, e_fake,      "Mean Energy (Fake)",   "E_fake",     "tomato")
    _plot(axes[1, 0], epochs, e_real_var,   "Energy Variance (Real)", "E_real_var", "forestgreen")
    _plot(axes[1, 1], epochs, e_fake_var,   "Energy Variance (Fake)", "E_fake_var", "firebrick")
    _plot(axes[1, 2], epochs, gap,          "Energy Gap (Real-Fake)", "Gap",        "mediumpurple")

    ax_fid = axes[2, 0]
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

    ax_precision = axes[2, 1]
    if precision:
        ax_precision.plot(pr_epochs, precision, color="slateblue", linewidth=2, marker="o", markersize=4)
        ax_precision.set_title("Precision", fontweight="bold")
        ax_precision.set_xlabel("Epoch")
        ax_precision.set_ylabel("Precision")
        ax_precision.grid(True, alpha=0.3)
    else:
        ax_precision.text(0.5, 0.5, "Precision not computed", ha="center", va="center",
                          transform=ax_precision.transAxes, fontsize=12)
        ax_precision.set_title("Precision")

    ax_recall = axes[2, 2]
    if recall:
        ax_recall.plot(pr_epochs, recall, color="crimson", linewidth=2, marker="o", markersize=4)
        ax_recall.set_title("Recall", fontweight="bold")
        ax_recall.set_xlabel("Epoch")
        ax_recall.set_ylabel("Recall")
        ax_recall.grid(True, alpha=0.3)
    else:
        ax_recall.text(0.5, 0.5, "Recall not computed", ha="center", va="center",
                       transform=ax_recall.transAxes, fontsize=12)
        ax_recall.set_title("Recall")

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
    if Path(path).name != "latest.pt":
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
