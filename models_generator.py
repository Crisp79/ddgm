import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # (B, 100) → (B, 512, 4, 4)
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True),

            nn.Unflatten(1, (512, 4, 4)),

            # (4x4 → 8x8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (8x8 → 16x16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (16x16 → 32x32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # (32x32 → 64x64)
            nn.ConvTranspose2d(64, channels, 4, 2, 1),

            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model
    G = Generator().to(device)

    # sample noise
    z = torch.randn(8, 100).to(device)

    # generate images
    fake = G(z)

    # move to cpu
    fake = fake.detach().cpu()

    # visualize first image
    img = fake[0].permute(1, 2, 0)   # C,H,W → H,W,C
    img = (img + 1) / 2              # [-1,1] → [0,1]

    plt.imshow(img)
    plt.axis("off")
    plt.show()