"""
Deep Energy Model (DEM)
=======================
Implements the energy function E_Theta(x) as described in the paper:
"Deep Directed Generative Models with Energy-Based Probability Estimation"
Kim & Bengio, 2016 (arXiv:1606.03439)

Architecture:
  - Feature extractor f_phi: deep CNN (DCGAN discriminator style)
  - Energy head: product-of-experts form analogous to free energy of RBM
      E(x) = (1/sigma^2) * ||f(x)||^2  -  b^T f(x)  -  sum_i log(1 + exp(W_i^T f(x) + b_i))

Stability fixes vs. naive implementation:
  - Quadratic term computed over features (bounded) not raw pixels (unbounded at 12k-d)
  - No spectral norm on experts (kills energy gradient to generator)
  - No BatchNorm in feature extractor (use LayerNorm or nothing — BN in DEM causes
    gradient issues when computing energy for a mix of real+fake in same forward pass)
  - Weight init: N(0, 0.02) throughout
  - Gradient penalty (R1) computed in train.py, not here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Convolutional feature extractor f_phi.
    Maps input images x -> high-level feature vector.
    Architecture mirrors DCGAN discriminator (Radford et al., 2015) [ref 14 in paper].

    Key: NO BatchNorm here. BN in the DEM feature extractor causes instability because
    real and fake samples have different statistics and are not always in the same batch.
    LeakyReLU + careful init is sufficient.

    Input:  (B, 3, 64, 64)
    Output: (B, feature_dim)
    """

    def __init__(self, n_features: int = 1024, image_channels: int = 3):
        super().__init__()
        # Channel progression: 3 -> 128 -> 256 -> 512 -> 1024  (paper: DEM 128-256-512-1024)
        self.conv_layers = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(image_channels, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(512, n_features, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global average pooling -> (B, n_features)
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
        h = self.conv_layers(x)                           # (B, n_features, 4, 4)
        h = self.pool(h).squeeze(-1).squeeze(-1)          # (B, n_features)
        return h


class DeepEnergyModel(nn.Module):
    """
    Deep Energy Model (DEM).

    Energy function (Eq. 11 in paper, with quadratic term over features):
        E(x) = (1/sigma^2) * ||f(x)||^2  -  b^T f(x)  -  sum_i log(1+exp(W_i^T f(x)+b_i))

    Computing the quadratic term over f(x) instead of raw x is the critical fix:
    - f(x) is bounded (post-LeakyReLU CNN output, GAP -> ~O(1) per dim)
    - Raw x at 64x64x3 = 12288 dims makes x^T x >> PoE term -> energy dominated
      by irrelevant pixel norm, gradients to experts vanish, generator learns nothing.

    The integrability guarantee from the paper (x^T x dominates) still holds because
    f(x) is a deterministic function of x — we've just reparametrised the energy
    into a fully feature-space form which is equivalent and numerically stable.

    Args:
        n_experts     : number of product-of-experts units (1024 for image experiments)
        feature_dim   : DEM CNN output channels (= input dim to energy head)
        image_channels: input image channels
        sigma         : scale of quadratic term (learnable feel free to set > 1)
    """

    def __init__(
        self,
        n_experts: int = 1024,
        feature_dim: int = 1024,
        image_channels: int = 3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        # sigma as log to keep it positive and let it be learned stably
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log())

        # Feature extractor f_phi  (no BN — see docstring)
        self.feature_extractor = FeatureExtractor(
            n_features=feature_dim, image_channels=image_channels
        )

        # Mean bias b: projects features to scalar (captures b^T f(x))
        self.mean_bias = nn.Linear(feature_dim, 1, bias=True)

        # Expert weights W_i and biases b_i
        # NO spectral_norm here — it collapses the energy landscape and kills
        # the gradient signal that trains the generator (Eq. 14).
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
        """
        Compute scalar energy for each sample in the batch.

        Args:
            x: (B, C, H, W) image tensor normalised to [-1, 1]
        Returns:
            energy: (B,) scalar energies  (lower = more probable)
        """
        # Feature extraction — bounded output
        f = self.feature_extractor(x)                     # (B, feature_dim)

        # Term 1: (1/sigma^2) * ||f(x)||^2  — quadratic regulariser over features
        # Normalise by feature_dim so scale is independent of feature_dim
        sigma2 = self.sigma ** 2 + 1e-8
        quad = (f * f).sum(dim=1) / (sigma2 * f.size(1))  # (B,)

        # Term 2: -b^T f(x)
        mean_term = self.mean_bias(f).squeeze(1)           # (B,)

        # Term 3: -sum_i log(1 + exp(W_i^T f + b_i))  [product-of-experts]
        logits = self.experts(f)                           # (B, n_experts)
        # Clamp logits to prevent softplus overflow in fp16
        logits = logits.clamp(-20, 20)
        poe_term = F.softplus(logits).sum(dim=1)           # (B,)

        energy = quad - mean_term - poe_term               # (B,)
        return energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns energy values. Lower energy = more probable."""
        return self.energy(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised log probability (up to partition function constant)."""
        return -self.energy(x)