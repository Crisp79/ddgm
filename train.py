import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models_energy import DeepEnergyModel
from models_generator import DeepGenerativeModel
from utils import (
    MetricLogger,
    save_samples,
    plot_training_metrics,
    save_checkpoint,
    load_checkpoint,
    InceptionFeatureExtractor,
    compute_fid,
    compute_precision_recall,
)
from dataset import get_celeba_loader


CONFIG = {
    "data_root": "./data",
    "subset_size": 20_000,
    "image_size": 64,
    "batch_size": 64,
    "num_workers": 4,
    "n_experts": 1024,
    "feature_dim": 1024,
    "latent_dim": 100,
    "prior": "normal",
    "sigma": 1.0,
    "epochs": 100,
    "lr_e": 1e-4,
    "lr_g": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "wd_g": 1e-4,
    "lambda_H": 1e-3,
    "r1_gamma": 10.0,
    "r1_every": 4,
    "margin": -1.0,
    "n_gen_steps": 1,
    "clip_grad_e": 5.0,
    "clip_grad_g": 1.0,
    "sample_every": 5,
    "save_checkpoint_every": 5,
    "n_samples": 64,
    "fid_every": 5,
    "fid_n_samples": 2000,
    "log_interval": 50,
    "output_dir": "./outputs",
    "seed": 42,
    "amp": True,
    "resume": None,
    "device": "auto",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DEM + DGM on CelebA")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sample_every", type=int, default=None)
    parser.add_argument("--save_checkpoint_every", type=int, default=None)
    parser.add_argument("--fid_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr_e", type=float, default=None)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lambda_H", type=float, default=None)
    parser.add_argument("--r1_gamma", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_args(cfg, args):
    for key in [
        "epochs",
        "sample_every",
        "save_checkpoint_every",
        "fid_every",
        "batch_size",
        "subset_size",
        "data_root",
        "output_dir",
        "lr_e",
        "lr_g",
        "lambda_H",
        "r1_gamma",
        "margin",
        "resume",
        "seed",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.no_amp:
        cfg["amp"] = False
    return cfg


def r1_gradient_penalty(dem, real_imgs: torch.Tensor) -> torch.Tensor:
    real_imgs = real_imgs.detach().requires_grad_(True)
    e_real = dem(real_imgs).sum()
    grads = torch.autograd.grad(
        outputs=e_real,
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = grads.pow(2).view(real_imgs.size(0), -1).sum(dim=1).mean()
    return penalty


def train_dem_step(dem, dgm, real_imgs, scaler_e, opt_e, cfg, device, batch_idx):
    B = real_imgs.size(0)
    apply_r1 = batch_idx % cfg["r1_every"] == 0

    opt_e.zero_grad(set_to_none=True)

    with autocast(
        device_type=device.type, enabled=cfg["amp"] and device.type == "cuda"
    ):
        e_real = dem(real_imgs)

        with torch.no_grad():
            z = dgm.sample_z(B, device)
            fake_imgs = dgm(z)

        e_fake = dem(fake_imgs.detach())

        margin = cfg.get("margin", -1.0)
        if margin > 0:
            loss_e = e_real.mean() + torch.nn.functional.relu(margin - e_fake).mean()
        else:
            loss_e = e_real.mean() - e_fake.mean()

    scaler_e.scale(loss_e).backward()

    if apply_r1:
        penalty = r1_gradient_penalty(dem, real_imgs)
        r1_loss = (cfg["r1_gamma"] / 2.0) * penalty * cfg["r1_every"]
        scaler_e.scale(r1_loss).backward()

    if cfg["clip_grad_e"] > 0:
        scaler_e.unscale_(opt_e)
        nn.utils.clip_grad_norm_(dem.parameters(), cfg["clip_grad_e"])

    scaler_e.step(opt_e)
    scaler_e.update()

    return (
        loss_e.item(),
        e_real.mean().item(),
        e_fake.mean().item(),
        e_real.var(unbiased=False).item(),
        e_fake.var(unbiased=False).item(),
    )


def train_dgm_step(dem, dgm, scaler_g, opt_g, cfg, device, batch_size):
    opt_g.zero_grad(set_to_none=True)
    with autocast(
        device_type=device.type, enabled=cfg["amp"] and device.type == "cuda"
    ):
        z = dgm.sample_z(batch_size, device)
        fake_imgs = dgm(z)

        e_fake_for_g = dem(fake_imgs)
        gen_energy_loss = e_fake_for_g.mean()

        entropy_reg = dgm.entropy_regularizer()

        loss_g = gen_energy_loss + cfg["lambda_H"] * entropy_reg

    scaler_g.scale(loss_g).backward()
    if cfg["clip_grad_g"] > 0:
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(dgm.parameters(), cfg["clip_grad_g"])
    scaler_g.step(opt_g)
    scaler_g.update()

    return loss_g.item()


def evaluate_fid(inception, dgm, loader, n_samples, fid_device, gen_device):
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
    precision, recall = compute_precision_recall(real_feats, fake_feats)
    print(f"  [FID] = {fid:.2f}")
    print(f"  [PR ] precision={precision:.3f} recall={recall:.3f}")
    return fid, precision, recall


def train(cfg: dict):

    torch.manual_seed(cfg["seed"])

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = get_celeba_loader(
        subset_size=cfg["subset_size"],
        batch_size=cfg["batch_size"],
        data_path="./data/celeba",
    )

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

    opt_e = torch.optim.Adam(
        dem.parameters(),
        lr=cfg["lr_e"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )

    opt_g = torch.optim.Adam(
        dgm.parameters(),
        lr=cfg["lr_g"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["wd_g"],
    )

    amp_enabled = cfg["amp"] and device.type == "cuda"
    scaler_e = GradScaler(device=device.type, enabled=amp_enabled)
    scaler_g = GradScaler(device=device.type, enabled=amp_enabled)

    start_epoch = 0
    if cfg["resume"] and Path(cfg["resume"]).exists():
        start_epoch = load_checkpoint(cfg["resume"], dem, dgm, opt_e, opt_g, device)

    logger = MetricLogger(str(output_dir / "metrics.csv"))

    inception = None
    fid_device = device if device.type == "cuda" else torch.device("cpu")

    dem.train()
    dgm.train()

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        t0 = time.time()
        epoch_e_loss = 0.0
        epoch_g_loss = 0.0
        epoch_e_real = 0.0
        epoch_e_fake = 0.0
        epoch_e_real_var = 0.0
        epoch_e_fake_var = 0.0
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

            loss_e, e_real, e_fake, e_real_var, e_fake_var = train_dem_step(
                dem, dgm, real_imgs, scaler_e, opt_e, cfg, device, batch_idx
            )

            loss_g = 0.0
            for _ in range(cfg["n_gen_steps"]):
                loss_g += train_dgm_step(dem, dgm, scaler_g, opt_g, cfg, device, B)
            loss_g /= cfg["n_gen_steps"]

            epoch_e_loss += loss_e
            epoch_g_loss += loss_g
            epoch_e_real += e_real
            epoch_e_fake += e_fake
            epoch_e_real_var += e_real_var
            epoch_e_fake_var += e_fake_var
            n_batches += 1

            progress.set_postfix(
                E_loss=f"{loss_e:+.4f}",
                G_loss=f"{loss_g:.4f}",
                E_real=f"{e_real:.4f}",
                E_fake=f"{e_fake:.4f}",
            )

        progress.close()

        avg_e_loss = epoch_e_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        avg_e_real = epoch_e_real / n_batches
        avg_e_fake = epoch_e_fake / n_batches
        avg_e_real_var = epoch_e_real_var / n_batches
        avg_e_fake_var = epoch_e_fake_var / n_batches
        gap = avg_e_real - avg_e_fake

        fid = float("nan")
        precision = float("nan")
        recall = float("nan")
        if epoch % cfg["fid_every"] == 0:
            if inception is None:
                print(f"  [FID] Loading InceptionV3 on {fid_device}...")
                inception = InceptionFeatureExtractor(torch.device("cpu"))

            if fid_device.type == "cuda":
                inception = inception.to(fid_device)

            fid, precision, recall = evaluate_fid(
                inception,
                dgm,
                loader,
                cfg["fid_n_samples"],
                fid_device=fid_device,
                gen_device=device,
            )

            if fid_device.type == "cuda":
                inception = inception.to(torch.device("cpu"))
                gc.collect()
                torch.cuda.empty_cache()

        elapsed = time.time() - t0

        metrics = {
            "e_loss": avg_e_loss,
            "g_loss": avg_g_loss,
            "e_real": avg_e_real,
            "e_fake": avg_e_fake,
            "e_real_var": avg_e_real_var,
            "e_fake_var": avg_e_fake_var,
            "gap": gap,
            "fid": fid if not (fid != fid) else "nan",
            "precision": precision if not (precision != precision) else "nan",
            "recall": recall if not (recall != recall) else "nan",
        }
        logger.log(epoch, metrics, elapsed)

        if epoch % cfg["sample_every"] == 0:
            save_samples(dgm, cfg["n_samples"], device, str(output_dir), epoch)

        ckpt_data = {
            "epoch": epoch,
            "dem": dem.state_dict(),
            "dgm": dgm.state_dict(),
            "opt_e": opt_e.state_dict(),
            "opt_g": opt_g.state_dict(),
            "scaler_e": scaler_e.state_dict(),
            "scaler_g": scaler_g.state_dict(),
            "cfg": cfg,
        }
        if epoch % cfg["save_checkpoint_every"] == 0:
            save_checkpoint(
                ckpt_data,
                str(output_dir / "checkpoints" / f"ckpt_epoch_{epoch:04d}.pt"),
            )
        save_checkpoint(
            ckpt_data,
            str(output_dir / "checkpoints" / "latest.pt"),
        )
        print(f"Epoch: {epoch} done")

    print("\n[Training complete] Saving final samples and metrics plot...")
    save_samples(dgm, cfg["n_samples"], device, str(output_dir), epoch=cfg["epochs"])
    plot_training_metrics(logger, str(output_dir))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    cfg = apply_args(CONFIG, args)

    print("=" * 60)
    print("  DEM + DGM Configuration")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k:<20}: {v}")
    print("=" * 60 + "\n")

    train(cfg)
