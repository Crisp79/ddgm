import math
import torch
import torch.nn as nn


class DeepGenerativeModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        n_features: int = 1024,
        image_channels: int = 3,
        prior: str = "normal",
    ):
        super().__init__()
        assert prior in ("uniform", "normal"), "prior must be 'uniform' or 'normal'"
        self.latent_dim = latent_dim
        self.prior = prior

        self.project = nn.Sequential(
            nn.Linear(latent_dim, n_features * 4 * 4, bias=False),
            nn.BatchNorm1d(n_features * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.n_features = n_features

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                n_features,
                n_features // 2,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                n_features // 2,
                n_features // 4,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                n_features // 4,
                n_features // 8,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_features // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                n_features // 8,
                image_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def sample_z(self, n: int, device: torch.device) -> torch.Tensor:
        if self.prior == "uniform":
            return torch.empty(n, self.latent_dim, device=device).uniform_(-1.0, 1.0)
        else:
            return torch.randn(n, self.latent_dim, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        h = self.project(z)
        h = h.view(z.size(0), self.n_features, 4, 4)
        x = self.deconv_layers(h)
        return x

    def generate(self, n: int, device: torch.device) -> torch.Tensor:
        z = self.sample_z(n, device)
        return self(z)

    def entropy_regularizer(self) -> torch.Tensor:

        neg_entropy = torch.tensor(0.0)
        two_e_pi = 2.0 * math.e * math.pi
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                sigma = m.weight

                h = 0.5 * torch.log(two_e_pi * sigma.pow(2) + 1e-8)
                neg_entropy = neg_entropy - h.sum()
        return neg_entropy.to(next(self.parameters()).device)
