"""
train.py
========
Training loop for Deep Energy Model (DEM) + Deep Generative Model (DGM).
Based on: "Deep Directed Generative Models with Energy-Based Probability Estimation"
          Kim & Bengio, 2016 (arXiv:1606.03439)

Usage:
    python train.py [--config config.yaml]  OR  edit CONFIG dict below directly.

Hardware target: Intel i7-12700H + NVIDIA RTX 3060 (6 GB VRAM)
  - Batch size 64 works comfortably within 6 GB.
  - Mixed precision (AMP) is enabled by default for ~2x speed boost.
  - FID is computed every `fid_every` epochs with a small sample (~2k) to stay fast.

Key training steps per batch (Eq. 7, 9, 13, 14 in paper):
  1. DEM update (positive + negative phase):
       L_E = E_{x+~P_D}[E(x+)] - E_{x-~P_phi}[E(x-)]
       -> Decrease energy of real data, increase energy of generated data.
  2. DGM update:
       L_G = E_{z~P(z)}[E(G(z))]  +  lambda_H * H_reg(G)
       -> Generator pushes samples toward low energy regions (Eq. 14).
       -> Entropy regularizer prevents mode collapse (Eq. 15).
"""

import argparse
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from models_energy import DeepEnergyModel
from models_generator import DeepGenerativeModel
from utils import (
    MetricLogger,
    save_samples,
    plot_training_metrics,
    save_checkpoint,
    load_checkpoint,
    InceptionFeatureExtractor,
    collect_inception_features,
    compute_fid,
)
from dataset import get_celeba_loader
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit these to customise the run)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "data_root":     "./data",          # root folder containing 'celeba/'
    "subset_size":   20_000,            # number of CelebA images to use
    "image_size":    64,
    "batch_size":    64,                # safe for RTX 3060 6 GB
    "num_workers":   4,                 # keep ≤ 4 on laptop CPUs

    # ── Model ─────────────────────────────────────────────────────────────────
    "n_experts":     1024,              # number of product-of-experts units (paper Sec. 4)
    "feature_dim":   1024,              # DEM/DGM feature map channels
    "latent_dim":    100,               # z dimensionality (paper Sec. 4)
    "prior":         "normal",          # 'normal' or 'uniform'
    "sigma":         1.0,               # DEM global variance

    # ── Training ──────────────────────────────────────────────────────────────
    "epochs":        100,               # total training epochs (set via CLI or edit here)
    "lr_e":          2e-4,              # Adam lr for DEM  (DCGAN-style)
    "lr_g":          2e-4,              # Adam lr for DGM
    "beta1":         0.5,               # Adam beta1 (DCGAN recommendation)
    "beta2":         0.999,
    "lambda_H":      1e-3,              # entropy regularizer weight (Eq. 15)
    "n_gen_steps":   1,                 # DGM updates per DEM update
    "clip_grad":     1.0,               # gradient clipping (stability)

    # ── Sampling & logging ────────────────────────────────────────────────────
    "sample_every":  5,                 # save sample grid every N epochs
    "n_samples":     64,                # images per sample grid
    "fid_every":     5,                 # compute FID every N epochs (slow; set high to skip)
    "fid_n_samples": 2000,              # samples for FID (2k is fast; 10k is more accurate)
    "log_interval":  50,                # print loss every N batches
    "output_dir":    "./outputs",

    # ── Misc ──────────────────────────────────────────────────────────────────
    "seed":          42,
    "amp":           True,              # mixed precision (recommended for RTX 3060)
    "resume":        None,              # path to checkpoint .pt to resume from
    "device":        "auto",            # 'auto' | 'cuda' | 'cpu'
}
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train DEM + DGM on CelebA")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--sample_every", type=int,   default=None)
    parser.add_argument("--fid_every",    type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--subset_size",  type=int,   default=None)
    parser.add_argument("--data_root",    type=str,   default=None)
    parser.add_argument("--output_dir",   type=str,   default=None)
    parser.add_argument("--lr_e",         type=float, default=None)
    parser.add_argument("--lr_g",         type=float, default=None)
    parser.add_argument("--lambda_H",     type=float, default=None)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--no_amp",       action="store_true")
    parser.add_argument("--seed",         type=int,   default=None)
    return parser.parse_args()


def apply_args(cfg, args):
    """Override CONFIG with any CLI arguments that were explicitly set."""
    for key in ["epochs", "sample_every", "fid_every", "batch_size",
                "subset_size", "data_root", "output_dir", "lr_e", "lr_g",
                "lambda_H", "resume", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.no_amp:
        cfg["amp"] = False
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Training step helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_dem_step(dem, dgm, real_imgs, scaler, opt_e, cfg, device):
    """
    Update DEM via Eq. 7 + 9 in paper:
        L_E = E[E(x+)]  -  E[E(x-)]
    Positive phase: push energy DOWN on real data.
    Negative phase: push energy UP on generated samples.
    Returns scalar loss value and mean energies.
    """
    B = real_imgs.size(0)

    opt_e.zero_grad(set_to_none=True)
    with autocast(enabled=cfg["amp"]):
        # Positive phase — real images
        e_real = dem(real_imgs)                           # (B,)

        # Negative phase — samples from generator (no grad through G here)
        with torch.no_grad():
            z = dgm.sample_z(B, device)
            fake_imgs = dgm(z)

        e_fake = dem(fake_imgs.detach())                  # (B,)

        # Loss: want E_real < E_fake  =>  minimise E_real - E_fake
        # (equivalent to MLE gradient Eq. 7)
        loss_e = e_real.mean() - e_fake.mean()

    scaler.scale(loss_e).backward()
    if cfg["clip_grad"] > 0:
        scaler.unscale_(opt_e)
        nn.utils.clip_grad_norm_(dem.parameters(), cfg["clip_grad"])
    scaler.step(opt_e)
    scaler.update()

    return (
        loss_e.item(),
        e_real.mean().item(),
        e_fake.mean().item(),
    )


def train_dgm_step(dem, dgm, scaler, opt_g, cfg, device, batch_size):
    """
    Update DGM via Eq. 13 + 14 in paper:
        L_G = E_z[E_Theta(G(z))]  +  lambda_H * (-H_approx)
    The gradient of -log P_Theta(G(z)) = E_Theta(G(z)) w.r.t. phi is
    computed by backprop through both G and E (Eq. 14).
    Entropy regularizer (Eq. 15) is added to encourage diversity.
    Returns scalar loss.
    """
    opt_g.zero_grad(set_to_none=True)
    with autocast(enabled=cfg["amp"]):
        z = dgm.sample_z(batch_size, device)
        fake_imgs = dgm(z)                        # (B, C, H, W)

        # Term 1: push generated samples toward low energy regions
        e_fake_for_g = dem(fake_imgs)             # (B,) — grad flows through G
        gen_energy_loss = e_fake_for_g.mean()

        # Term 2: entropy regularizer (Eq. 15) — maximise diversity
        entropy_reg = dgm.entropy_regularizer()   # negative entropy to minimise

        loss_g = gen_energy_loss + cfg["lambda_H"] * entropy_reg

    scaler.scale(loss_g).backward()
    if cfg["clip_grad"] > 0:
        scaler.unscale_(opt_g)
        nn.utils.clip_grad_norm_(dgm.parameters(), cfg["clip_grad"])
    scaler.step(opt_g)
    scaler.update()

    return loss_g.item()


# ──────────────────────────────────────────────────────────────────────────────
# FID evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_fid(inception, dgm, loader, n_samples, fid_device, gen_device):
    """Compute FID between real data and generated samples."""
    print(f"  [FID] Collecting {n_samples} real features...")
    real_feats = []
    collected = 0
    for batch in loader:
        if collected >= n_samples:
            break
        imgs = batch[0].to(fid_device)
        with torch.no_grad():
            f = inception(imgs).cpu().numpy()
        real_feats.append(f)
        collected += f.shape[0]

    import numpy as np
    real_feats = np.concatenate(real_feats, axis=0)[:n_samples]

    print(f"  [FID] Generating {n_samples} fake features...")
    fake_feats = []
    generated = 0
    dgm.eval()
    while generated < n_samples:
        bs = min(64, n_samples - generated)
        with torch.no_grad():
            imgs = dgm.generate(bs, gen_device)
            imgs = imgs.to(fid_device)
        f = inception(imgs).cpu().numpy()
        fake_feats.append(f)
        generated += bs
    dgm.train()
    fake_feats = np.concatenate(fake_feats, axis=0)[:n_samples]

    fid = compute_fid(real_feats, fake_feats)
    print(f"  [FID] = {fid:.2f}")
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = get_celeba_loader(
        subset_size=cfg["subset_size"],
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
        num_workers=cfg["num_workers"],
        data_path="./data/celeba"
    )

    # ── Models ────────────────────────────────────────────────────────────────
    dem = DeepEnergyModel(
        n_experts=cfg["n_experts"],
        feature_dim=cfg["feature_dim"],
        sigma=cfg["sigma"],
    ).to(device)

    dgm = DeepGenerativeModel(
        latent_dim=cfg["latent_dim"],
        n_features=cfg["feature_dim"],
        prior=cfg["prior"],
    ).to(device)

    n_params_dem = sum(p.numel() for p in dem.parameters() if p.requires_grad)
    n_params_dgm = sum(p.numel() for p in dgm.parameters() if p.requires_grad)
    print(f"[Models] DEM params: {n_params_dem:,} | DGM params: {n_params_dgm:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_e = torch.optim.Adam(dem.parameters(), lr=cfg["lr_e"],
                              betas=(cfg["beta1"], cfg["beta2"]))
    opt_g = torch.optim.Adam(dgm.parameters(), lr=cfg["lr_g"],
                              betas=(cfg["beta1"], cfg["beta2"]))

    scaler = GradScaler(enabled=cfg["amp"])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if cfg["resume"] and Path(cfg["resume"]).exists():
        start_epoch = load_checkpoint(cfg["resume"], dem, dgm, opt_e, opt_g, device)

    # ── Logging ───────────────────────────────────────────────────────────────
    logger = MetricLogger(str(output_dir / "metrics.csv"))

    # ── Inception for FID (lazy init) ─────────────────────────────────────────
    inception = None
    fid_device = torch.device("cpu")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training DEM + DGM for {cfg['epochs']} epochs")
    print(f"  Sample every: {cfg['sample_every']} epochs")
    print(f"  FID every:    {cfg['fid_every']} epochs ({cfg['fid_n_samples']} samples)")
    print(f"{'='*60}\n")

    dem.train()
    dgm.train()

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        t0 = time.time()
        epoch_e_loss = 0.0
        epoch_g_loss = 0.0
        epoch_e_real = 0.0
        epoch_e_fake = 0.0
        n_batches = 0

        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch:>3}/{cfg['epochs']}",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, (real_imgs, _) in enumerate(progress):
            real_imgs = real_imgs.to(device, non_blocking=True)
            B = real_imgs.size(0)

            # ── 1. Update DEM ────────────────────────────────────────────────
            loss_e, e_real, e_fake = train_dem_step(
                dem, dgm, real_imgs, scaler, opt_e, cfg, device
            )

            # ── 2. Update DGM (n_gen_steps times) ───────────────────────────
            loss_g = 0.0
            for _ in range(cfg["n_gen_steps"]):
                loss_g += train_dgm_step(dem, dgm, scaler, opt_g, cfg, device, B)
            loss_g /= cfg["n_gen_steps"]

            epoch_e_loss += loss_e
            epoch_g_loss += loss_g
            epoch_e_real += e_real
            epoch_e_fake += e_fake
            n_batches += 1

            progress.set_postfix(
                E_loss=f"{loss_e:+.4f}",
                G_loss=f"{loss_g:.4f}",
                E_real=f"{e_real:.4f}",
                E_fake=f"{e_fake:.4f}",
            )

        progress.close()

        # ── Epoch-level averages ──────────────────────────────────────────────
        avg_e_loss = epoch_e_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        avg_e_real = epoch_e_real / n_batches
        avg_e_fake = epoch_e_fake / n_batches
        gap        = avg_e_real - avg_e_fake   # ideally negative & stable

        # ── FID ──────────────────────────────────────────────────────────────
        fid = float("nan")
        if epoch % cfg["fid_every"] == 0:
            if inception is None:
                print("  [FID] Loading InceptionV3 on CPU...")
                inception = InceptionFeatureExtractor(fid_device)
            fid = evaluate_fid(
                inception,
                dgm,
                loader,
                cfg["fid_n_samples"],
                fid_device=fid_device,
                gen_device=device,
            )

        elapsed = time.time() - t0

        # ── Log & print ───────────────────────────────────────────────────────
        metrics = {
            "e_loss": avg_e_loss,
            "g_loss": avg_g_loss,
            "e_real": avg_e_real,
            "e_fake": avg_e_fake,
            "gap":    gap,
            "fid":    fid if not (fid != fid) else "nan",  # NaN check
        }
        logger.log(epoch, metrics, elapsed)

        # ── Samples ───────────────────────────────────────────────────────────
        if epoch % cfg["sample_every"] == 0:
            save_samples(dgm, cfg["n_samples"], device, str(output_dir), epoch)

        # ── Checkpoint ───────────────────────────────────────────────────────
        save_checkpoint(
            {
                "epoch": epoch,
                "dem":   dem.state_dict(),
                "dgm":   dgm.state_dict(),
                "opt_e": opt_e.state_dict(),
                "opt_g": opt_g.state_dict(),
                "cfg":   cfg,
            },
            str(output_dir / "checkpoints" / f"ckpt_epoch_{epoch:04d}.pt"),
        )
        # Keep latest checkpoint separately for easy resume
        save_checkpoint(
            {
                "epoch": epoch,
                "dem":   dem.state_dict(),
                "dgm":   dgm.state_dict(),
                "opt_e": opt_e.state_dict(),
                "opt_g": opt_g.state_dict(),
                "cfg":   cfg,
            },
            str(output_dir / "checkpoints" / "latest.pt"),
        )

    # ── Post-training: final samples + metrics plot ───────────────────────────
    print("\n[Training complete] Saving final samples and metrics plot...")
    save_samples(dgm, cfg["n_samples"], device, str(output_dir), epoch=cfg["epochs"])
    plot_training_metrics(logger, str(output_dir))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = apply_args(CONFIG, args)

    print("=" * 60)
    print("  DEM + DGM Configuration")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k:<20}: {v}")
    print("=" * 60 + "\n")

    train(cfg)