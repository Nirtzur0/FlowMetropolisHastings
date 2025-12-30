import torch
import torch.nn as nn
from typing import Tuple, Optional

class TransferMatrixMethod(nn.Module):
    """
    Differentiable Transfer Matrix Method (TMM) solver for multilayer thin films.
    Calculates Reflectance (R) and Transmittance (T) for a stack of layers.
    
    Assumptions:
    - Normal incidence (can be extended to angles).
    - Non-magnetic materials (mu=1).
    - Polarization generic (normal incidence s=p).
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                n_layers: torch.Tensor, 
                d_layers: torch.Tensor, 
                wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_layers: (Batch, N_layers) Refractive indices (complex or real).
                      Typically: [n_substrate, n_1, n_2, ..., n_ambient]
            d_layers: (Batch, N_layers) Physical thicknesses in nanometers.
                      Note: Substrate and Ambient thicknesses are ignored (treated as semi-infinite).
            wavelengths: (W,) Wavelengths in nanometers.
            
        Returns:
            R: (Batch, W) Reflectance spectrum.
        """
        # Ensure complex for Fresnel calc
        if not n_layers.is_complex():
            n_layers = n_layers.to(torch.complex64)
            
        # Add singleton dims for broadcasting:
        # Batch (B), Layers (L), Wavelengths (W)
        # n_layers: (B, L, 1)
        # d_layers: (B, L, 1)
        # wavelengths: (1, 1, W)
        
        n = n_layers.unsqueeze(-1) # (B, N, 1)
        d = d_layers.unsqueeze(-1) # (B, N, 1)
        lam = wavelengths.view(1, 1, -1) # (1, 1, W)
        
        batch_size = n.shape[0]
        # num_layers = n.shape[1] # Unused if we vectorize
        num_waves = lam.shape[2]
        
        # 1. Compute Layer Matrices for ALL layers at once (Vectorized)
        # Layers 1 to N-2 (Internal Layers)
        # We slice indices [1:-1]
        
        n_i = n[:, 1:-1, :] # (B, N_internal, 1)
        d_i = d[:, 1:-1, :] # (B, N_internal, 1)
        
        k0 = 2 * torch.pi / lam # (1, 1, W)
        phi = n_i * d_i * k0 # (B, N_int, W)
        
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        # Construct M_layer tensor: (B, N_int, W, 2, 2)
        # M = [[cos, -i/n sin], [-in sin, cos]]
        
        zeros = torch.zeros_like(phi)
        
        m11 = cos_phi
        m12 = -1j * sin_phi / (n_i + 1e-8)
        m21 = -1j * n_i * sin_phi
        m22 = cos_phi
        
        # We need these stacked into (..., 2, 2) matrices.
        # Current shape of m elements: (B, N_int, W)
        
        # Stack to (B, N_int, W, 2, 2)
        # Rows:
        row1 = torch.stack([m11, m12], dim=-1) # (..., 2)
        row2 = torch.stack([m21, m22], dim=-1) # (..., 2)
        M_all = torch.stack([row1, row2], dim=-2) # (B, N_int, W, 2, 2)
        
        # 2. Multiply Matrices
        # We need M_total = M_1 @ M_2 @ ... @ M_N
        # PyTorch doesn't have a built-in "matmul along dim" (reduce matmul).
        # But since N_layers is small (~10-50), a simple loop over the reduced dimension is cleaner than generic reduce.
        # Alternatively, assume we can reshape? No, order matters.
        
        # However, we can use the loop here but keeping (W) vectorized.
        # Previous code had W inside the matrix or permuted.
        
        # Let's define Accumulated M: (B, W, 2, 2)
        # Start as Identity
        M_total = torch.eye(2, dtype=torch.complex64, device=n.device)
        M_total = M_total.view(1, 1, 2, 2).expand(batch_size, num_waves, -1, -1).clone()
        
        # Rearrange M_all to (N_int, B, W, 2, 2) to iterate over layers easily
        M_all = M_all.permute(1, 0, 2, 3, 4)
        
        # Loop over layers (Vectorized over Batch and Wavelength now!)
        for i in range(M_all.shape[0]):
             M_layer = M_all[i] # (B, W, 2, 2)
             M_total = torch.matmul(M_total, M_layer)
             
        # 3. Apply Boundary Conditions
        # n_in (Ambient, index 0), n_sub (Substrate, index -1)
        n_in = n[:, 0, :]   # (B, 1)
        n_sub = n[:, -1, :] # (B, 1)
        
        # BC_sub: [1, n_sub]
        # (B, W, 2, 1)
        BC_sub = torch.zeros(batch_size, num_waves, 2, 1, dtype=torch.complex64, device=n.device)
        BC_sub[:, :, 0, 0] = 1.0
        BC_sub[:, :, 1, 0] = n_sub.expand(-1, num_waves)
        
        # Propagate to input
        # EH_in = M_total @ BC_sub
        EH_in = torch.matmul(M_total, BC_sub) # (B, W, 2, 1)
        
        E_in = EH_in[:, :, 0, 0] 
        H_in = EH_in[:, :, 1, 0]
        
        # Input Admittance
        Y_in = H_in / (E_in + 1e-9)
        
        # Reflectance
        n_in_sq = n_in.expand(-1, num_waves)
        r = (n_in_sq - Y_in) / (n_in_sq + Y_in)
        R = torch.abs(r)**2
        
        return R
