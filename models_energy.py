import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class EnergyModel(nn.Module):
    def __init__(self, in_channels=3, num_experts=128):
        super().__init__()

        self.feature_net = FeatureExtractor(in_channels)

        self.feature_dim = 512 * 4 * 4

        self.experts = nn.Linear(self.feature_dim, num_experts)

        self.linear = nn.Linear(64 * 64 * in_channels, 1)

    def forward(self, x):
        B = x.size(0)

        x_flat = x.view(B, -1)

        # term1 (scaled)
        term1 = (x_flat ** 2).mean(dim=1)

        # term2 (scaled)
        term2 = self.linear(x_flat).squeeze() / x_flat.size(1)

        # feature extractor
        f = self.feature_net(x)

        # stable expert term
        experts_out = self.experts(f)
        term3 = F.softplus(experts_out).mean(dim=1)

        energy = term1 - term2 - term3

        # clamp for stability
        energy = 10 * torch.tanh(energy / 10)

        return energy