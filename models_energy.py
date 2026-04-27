import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, n_features: int = 1024, image_channels: int = 3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                image_channels, 128, kernel_size=5, stride=2, padding=2, bias=True
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, n_features, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = n_features

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        return h


class DeepEnergyModel(nn.Module):
    def __init__(
        self,
        n_experts: int = 1024,
        feature_dim: int = 1024,
        image_channels: int = 3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts

        self.log_sigma = nn.Parameter(torch.tensor(sigma).log())

        self.feature_extractor = FeatureExtractor(
            n_features=feature_dim, image_channels=image_channels
        )

        self.mean_bias = nn.Linear(feature_dim, 1, bias=True)

        self.experts = nn.Linear(feature_dim, n_experts, bias=True)

        self._init_energy_head()

    def _init_energy_head(self):
        nn.init.normal_(self.mean_bias.weight, 0.0, 0.02)
        nn.init.zeros_(self.mean_bias.bias)
        nn.init.normal_(self.experts.weight, 0.0, 0.02)
        nn.init.zeros_(self.experts.bias)

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def energy(self, x: torch.Tensor) -> torch.Tensor:

        f = self.feature_extractor(x)

        sigma2 = self.sigma**2 + 1e-8
        quad = (f * f).sum(dim=1) / (sigma2 * f.size(1))

        mean_term = self.mean_bias(f).squeeze(1)

        logits = self.experts(f)

        logits = logits.clamp(-20, 20)
        poe_term = F.softplus(logits).sum(dim=1)

        energy = quad - mean_term - poe_term
        return energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.energy(x)
