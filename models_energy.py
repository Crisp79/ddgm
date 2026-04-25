"""
Deep Energy Model (DEM)
=======================
Implements the energy function E_Theta(x) as described in the paper:
"Deep Directed Generative Models with Energy-Based Probability Estimation"
Kim & Bengio, 2016 (arXiv:1606.03439)

Architecture:
  - Feature extractor f_phi: deep CNN (for image data)
  - Energy head: product-of-experts form analogous to free energy of RBM
      E(x) = (1/sigma^2) * x^T x  -  b^T x  -  sum_i log(1 + exp(W_i^T f(x) + b_i))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Convolutional feature extractor f_phi.
    Maps input images x -> high-level feature vector.
    Architecture mirrors DCGAN discriminator (Radford et al., 2015) [ref 14 in paper].
    Input:  (B, 3, 64, 64)
    Output: (B, feature_dim)
    """

    def __init__(self, n_features: int = 1024, image_channels: int = 3):
        super().__init__()
        # Channel progression: 3 -> 128 -> 256 -> 512 -> 1024  (paper: DEM 128-256-512-1024)
        self.conv_layers = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(image_channels, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(512, n_features, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(n_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global average pooling -> (B, n_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x)          # (B, n_features, 4, 4)
        h = self.pool(h).squeeze(-1).squeeze(-1)  # (B, n_features)
        return h


class DeepEnergyModel(nn.Module):
    """
    Deep Energy Model (DEM).

    Energy function (Eq. 11 in paper):
        E(x) = (1/sigma^2) * ||x||^2  -  b^T x  -  sum_i log(1 + exp(W_i^T f(x) + b_i))

    The last term is a product-of-experts over the feature space.
    Integrability w.r.t. x is guaranteed because x^T x dominates.

    Args:
        n_experts   : number of expert units (1024 for image experiments, paper Sec. 4)
        feature_dim : dimensionality of the feature extractor output
        image_channels: input image channels
        sigma       : global variance parameter (fixed)
    """

    def __init__(
        self,
        n_experts: int = 1024,
        feature_dim: int = 1024,
        image_channels: int = 3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.n_experts = n_experts

        # Feature extractor f_phi
        self.feature_extractor = FeatureExtractor(
            n_features=feature_dim, image_channels=image_channels
        )

        # Mean bias b (captures global mean of x)
        # Flattened image size for b: 3*64*64 = 12288
        # We use a learned scalar projection instead of full b^T x to keep memory tractable
        self.mean_bias = nn.Linear(feature_dim, 1, bias=True)

        # Expert weights W_i and biases b_i  (single linear layer = n_experts experts)
        self.experts = nn.Linear(feature_dim, n_experts, bias=True)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar energy for each sample in the batch.

        Args:
            x: (B, C, H, W) image tensor, assumed normalised to [-1, 1]
        Returns:
            energy: (B,) scalar energies
        """
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, C*H*W)

        # Term 1: (1/sigma^2) * ||x||^2
        quad = (1.0 / (self.sigma ** 2)) * (x_flat * x_flat).sum(dim=1)  # (B,)

        # Feature extraction
        f = self.feature_extractor(x)  # (B, feature_dim)

        # Term 2: -b^T x  (approximated via feature projection for tractability)
        mean_term = self.mean_bias(f).squeeze(1)  # (B,)

        # Term 3: -sum_i log(1 + exp(W_i^T f + b_i))  [product-of-experts]
        logits = self.experts(f)             # (B, n_experts)
        poe_term = F.softplus(logits).sum(dim=1)  # (B,)  softplus = log(1+exp(.))

        energy = quad - mean_term - poe_term  # (B,)
        return energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns energy values. Lower energy = more probable."""
        return self.energy(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log probability: log p(x) = -E(x) - log Z.
        Since log Z is constant w.r.t. parameters in the gradient, we return -E(x).
        """
        return -self.energy(x)