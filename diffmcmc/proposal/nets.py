import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels/time."""
    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        # Random weight matrix: (dim // 2,)
        # We want to map scalar t -> embed_dim vector
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B,)
        # x_proj: (B, dim//2)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResNetBlock(nn.Module):
    """
    ResNet block with Adaptive Group Normalization (conditionally scaled/shifted by time).
    """
    def __init__(self, dim, time_embed_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Linear(dim, dim)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim * 2) # shift & scale
        )
        
        self.norm2 = nn.GroupNorm(8, dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Add time conditioning
        # time_emb: (B, time_embed_dim)
        style = self.time_proj(time_emb) # (B, 2*dim)
        scale, shift = torch.chunk(style, 2, dim=1)
        
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.act2(h))
        return x + self.dropout(h)

class TimeResNet(nn.Module):
    """
    Residual Network for Velocity Field Estimation.
    """
    def __init__(self, dim, hidden_dim=128, time_embed_dim=64, num_layers=3):
        super().__init__()
        self.time_embed = GaussianFourierProjection(time_embed_dim, scale=1.0)
        
        # Initial projection
        self.head = nn.Linear(dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, time_embed_dim) for _ in range(num_layers)
        ])
        
        self.final = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x, t):
        # x: (B, D)
        # t: (B,) or scalar
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t = torch.ones(x.shape[0], device=x.device) * t
        elif t.ndim == 2:
            t = t.squeeze(1)
            
        # Ensure t is broadcast to batch if needed (though usually passed correctly)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
            
        emb = self.time_embed(t)
        
        h = self.head(x)
        for block in self.blocks:
            h = block(h, emb)
            
        return self.final(h)

class VelocityMLP(nn.Module):
    """
    Simple MLP velocity field v(x, t).
    Kept for backward compatibility and simple baselines.
    """
    def __init__(self, dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.dim = dim
        input_dim = dim + 1 
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t = torch.ones(x.shape[0], 1, device=x.device) * t
        elif t.ndim == 1:
            t = t.unsqueeze(1)
            
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
            
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)
