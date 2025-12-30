import torch
import torch.distributions as dist
from typing import Optional

class FunnelTarget:
    """
    Neal's Funnel distribution.
    
    Generative process:
    y ~ N(0, 3^2)
    x_i ~ N(0, exp(y/2)^2) for i in 1..D-1
    
    The log-probability is:
    log p(x, y) = log p(y) + sum_i log p(x_i | y)
    """
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.y_dist = dist.Normal(loc=0.0, scale=3.0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability density of the Funnel distribution.
        
        Args:
            x: shape (..., D) where x[..., 0] is y and x[..., 1:] are x_i components.
            
        Returns:
            log_prob: shape (...,)
        """
        # Ensure correct shape handling
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)
            
        # x is (..., D)
        # y is the first dimension component (index 0)
        # others are the rest (indices 1:)
        y = x[..., 0]      # (...,)
        others = x[..., 1:] # (..., D-1)
        
        # log p(y)
        log_prob_y = self.y_dist.log_prob(y)
        
        # Conditional distribution for others given y:
        # others ~ N(0, (exp(y/2))^2)
        # scale = exp(y/2)
        scale = torch.exp(y / 2.0).unsqueeze(-1) # (..., 1) matches others' last dim broadcasting
        
        others_dist = dist.Normal(loc=0.0, scale=scale)
        log_prob_others = others_dist.log_prob(others).sum(dim=-1) # Sum over D-1 dimensions
        
        return log_prob_y + log_prob_others
