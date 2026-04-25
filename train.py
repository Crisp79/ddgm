import torch
import torch.optim as optim
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

from dataset import get_celeba_loader
from models_generator import Generator
from models_energy import EnergyModel
from torchvision.utils import save_image


# =========================
# PLOTTING FUNCTION
# =========================
def plot_metrics(loss_E, loss_G, E_real, E_fake, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(loss_E) + 1))

    # ======================
    # SAVE CSV
    # ======================
    csv_path = os.path.join(save_dir, "metrics.csv")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # header
        writer.writerow(["epoch", "loss_E", "loss_G", "E_real", "E_fake", "gap"])

        # rows
        for i in range(len(epochs)):
            gap = E_fake[i] - E_real[i]
            writer.writerow([
                epochs[i],
                loss_E[i],
                loss_G[i],
                E_real[i],
                E_fake[i],
                gap
            ])

    # ======================
    # LOSS PLOT
    # ======================
    plt.figure()
    plt.plot(epochs, loss_E, label="Energy Loss")
    plt.plot(epochs, loss_G, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    # ======================
    # ENERGY PLOT
    # ======================
    plt.figure()
    plt.plot(epochs, E_real, label="E_real")
    plt.plot(epochs, E_fake, label="E_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("Energy Values")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "energy_plot.png"))
    plt.close()

    # ======================
    # ENERGY GAP PLOT
    # ======================
    gap = [f - r for f, r in zip(E_fake, E_real)]

    plt.figure()
    plt.plot(epochs, gap, label="Energy Gap")
    plt.xlabel("Epoch")
    plt.ylabel("Gap")
    plt.title("Energy Separation")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "energy_gap.png"))
    plt.close()


# =========================
# TRAIN FUNCTION
# =========================
def train(epochs=100,size=20000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # models
    G = Generator().to(device)
    E = EnergyModel().to(device)

    # optimizers
    opt_E = optim.Adam(E.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # data
    loader = get_celeba_loader("data/celeba", batch_size=64, subset_size=size)

    # outputs
    os.makedirs("outputs", exist_ok=True)

    # metric history
    loss_E_history = []
    loss_G_history = []
    E_real_history = []
    E_fake_history = []

    ema_gap = 0
    alpha = 0.9

    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        # epoch accumulators
        epoch_loss_E = 0
        epoch_loss_G = 0
        epoch_E_real = 0
        epoch_E_fake = 0
        count = 0

        for images, _ in loop:
            images = images + 0.01 * torch.randn_like(images)
            images = images.to(device)
            B = images.size(0)

            # ======================
            # Train Energy Model
            # ======================
            z = torch.randn(B, 256).to(device)
            fake = G(z)

            E_real = E(images)
            E_fake = E(fake.detach())

            margin = 4.0
            loss_E = torch.relu(E_real.mean() - E_fake.mean() + margin)

            reg = 0.001 * (E_real.pow(2).mean() + E_fake.pow(2).mean())
            loss_E = loss_E + reg

            opt_E.zero_grad()
            loss_E.backward()
            torch.nn.utils.clip_grad_norm_(E.parameters(), 1.0)
            opt_E.step()

            # ======================
            # Train Generator
            # ======================
            for _ in range(1):
                z = torch.randn(B, 256).to(device)
                fake = G(z)

                E_fake = E(fake)

                entropy = 0
                for m in G.modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                        entropy += torch.log(m.running_var + 1e-6).mean()

                if epoch < 5:
                    loss_G = E_fake.mean()
                else:
                    loss_G = E_fake.mean() - 0.01 * entropy

                opt_G.zero_grad()
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

            # ======================
            # NaN safety
            # ======================
            if torch.isnan(loss_E) or torch.isnan(loss_G):
                print("NaN detected, stopping...")
                return

            # ======================
            # accumulate metrics
            # ======================
            epoch_loss_E += loss_E.item()
            epoch_loss_G += loss_G.item()
            epoch_E_real += E_real.mean().item()
            epoch_E_fake += E_fake.mean().item()
            count += 1

            loop.set_postfix(
                loss_E=loss_E.item(),
                loss_G=loss_G.item()
            )

        # ======================
        # store epoch averages
        # ======================
        loss_E_history.append(epoch_loss_E / count)
        loss_G_history.append(epoch_loss_G / count)
        E_real_history.append(epoch_E_real / count)
        E_fake_history.append(epoch_E_fake / count)

        print(
            f"\nEpoch {epoch+1}: "
            f"E_real={E_real_history[-1]:.4f}, "
            f"E_fake={E_fake_history[-1]:.4f}"
        )

        gap = E_fake_history[-1] - E_real_history[-1]
        ema_gap = alpha * ema_gap + (1 - alpha) * gap
        print(f"gap: {gap:.3f}, ema_gap: {ema_gap:.3f}")

        # ======================
        # save samples
        # ======================
        save_image(
            (fake + 1) / 2,
            f"outputs/epoch_{epoch}.png",
            nrow=4
        )

    # ======================
    # plot metrics
    # ======================
    plot_metrics(
        loss_E_history,
        loss_G_history,
        E_real_history,
        E_fake_history
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train(epochs=40,size=20000)