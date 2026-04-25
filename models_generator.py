"""
Deep Generative Model (DGM)
============================
Implements the generator G_phi described in the paper:
"Deep Directed Generative Models with Energy-Based Probability Estimation"
Kim & Bengio, 2016 (arXiv:1606.03439)

The generator is a directed graphical model:
  z ~ P(z)   (simple prior, e.g. Uniform(-1,1) or Normal(0,1))
  x = G_phi(z)  (deterministic transformation via deep neural network)

Training objective (Eq. 10 / 13 in paper):
  min KL(P_phi(x) || P_Theta(x))
    = E_{x~P_phi}[-log P_Theta(x)] - H(P_phi(x))
    = E_z[E_Theta(G(z))]  +  entropy_regularizer

The entropy regularizer (Eq. 15) uses batch-normalisation scale params:
  H(P_phi) ≈ sum_i  (1/2) log(2*e*pi*sigma_ai^2)
which is maximised by encouraging BN scale params to be large (anti-weight-decay).

Architecture: DCGAN-style transposed convolution generator (Radford et al., 2015) [ref 14]
    z (latent_dim=100) -> 4x4 feature maps -> ... -> 64x64 RGB image
    Channel progression (DGM): 1024 -> 512 -> 256 -> 128  (paper Sec. 4)
"""

import math
import torch
import torch.nn as nn


class DeepGenerativeModel(nn.Module):
    """
    Deep Generative Model (DGM) / Generator G_phi.

    Args:
        latent_dim    : dimensionality of latent vector z (100 for image experiments)
        n_features    : base number of feature maps (1024 = start of upsampling)
        image_channels: output image channels (3 for RGB)
        prior         : 'uniform' ~ U(-1,1) or 'normal' ~ N(0,1)
    """

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

        # Project z -> (n_features, 4, 4) spatial feature maps
        self.project = nn.Sequential(
            nn.Linear(latent_dim, n_features * 4 * 4, bias=False),
            nn.BatchNorm1d(n_features * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.n_features = n_features

        # Transposed conv stack: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # Channel progression: 1024 -> 512 -> 256 -> 128 -> 3  (paper DGM: 1024-512-256-128)
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(n_features, n_features // 2, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(n_features // 2),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(n_features // 2, n_features // 4, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(n_features // 4),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(n_features // 4, n_features // 8, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(n_features // 8),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(n_features // 8, image_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """DCGAN-style weight initialisation: N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def sample_z(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample latent vectors from the prior P(z)."""
        if self.prior == "uniform":
            return torch.empty(n, self.latent_dim, device=device).uniform_(-1.0, 1.0)
        else:
            return torch.randn(n, self.latent_dim, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from latent codes.

        Args:
            z: (B, latent_dim) latent tensor
        Returns:
            x: (B, C, 64, 64) generated images in [-1, 1]
        """
        h = self.project(z)                              # (B, n_features*4*4)
        h = h.view(z.size(0), self.n_features, 4, 4)    # (B, n_features, 4, 4)
        x = self.deconv_layers(h)                        # (B, C, 64, 64)
        return x

    def generate(self, n: int, device: torch.device) -> torch.Tensor:
        """Convenience: sample z then generate."""
        z = self.sample_z(n, device)
        return self(z)

    def entropy_regularizer(self) -> torch.Tensor:
        """
        Approximate entropy of the generator distribution via BN scale parameters (Eq. 15):
            H(P_phi) ≈ sum_i  (1/2) * log(2*e*pi*sigma_ai^2)

        Maximising this (= minimising its negative) encourages diverse samples and
        prevents mode collapse.

        Returns:
            scalar tensor (negative entropy to be minimised)
        """
        neg_entropy = torch.tensor(0.0)
        two_e_pi = 2.0 * math.e * math.pi
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # sigma_ai is the learned scale parameter (weight) of BN
                sigma = m.weight  # (C,) or (features,)
                # H = 0.5 * log(2*e*pi*sigma^2)  per activation
                h = 0.5 * torch.log(two_e_pi * sigma.pow(2) + 1e-8)
                neg_entropy = neg_entropy - h.sum()  # we minimise -H
        return neg_entropy.to(next(self.parameters()).device)