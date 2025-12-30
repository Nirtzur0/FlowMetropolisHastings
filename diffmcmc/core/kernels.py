import abc
import torch
import numpy as np
from typing import Tuple, Callable

class AbstractKernel(abc.ABC):
    """Abstract base class for local MCMC kernels."""
    
    @abc.abstractmethod
    def propose(self, current_x: torch.Tensor, log_prob_fn: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propose a new state given the current state.
        
        Args:
            current_x: (D,) Tensor
            log_prob_fn: function mapping (D,) -> (1,)
            
        Returns:
             proposed_x: (D,) Tensor
             log_q_ratio: log q(x|x') - log q(x'|x) (backward - forward)
             correction: Any other MH correction term (usually 0 for symmetric)
        """
        pass

class RWMKernel(AbstractKernel):
    """Random Walk Metropolis Kernel with isotropic Gaussian proposal."""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        
    def propose(self, current_x: torch.Tensor, log_prob_fn: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x' = x + N(0, scale^2 I)
        noise = torch.randn_like(current_x) * self.scale
        proposed_x = current_x + noise
        
        # Symmetric proposal: q(x'|x) = q(x|x')
        # log_q_ratio = 0
        return proposed_x, torch.tensor(0.0, device=current_x.device)

class MALAKernel(AbstractKernel):
    """Metropolis-Adjusted Langevin Algorithm Kernel."""
    
    def __init__(self, step_size: float):
        self.step_size = step_size
        
    def propose(self, current_x: torch.Tensor, log_prob_fn: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gradient computation required. 
        # We need the gradient of log_prob at current_x.
        
        x = current_x.detach().clone().requires_grad_(True)
        lp = log_prob_fn(x)
        grad = torch.autograd.grad(lp, x)[0]
        
        dt = self.step_size ** 2
        # x' = x + (dt/2) * grad + sqrt(dt) * z
        noise = torch.randn_like(current_x)
        if torch.isnan(grad).any():
             # Fallback if gradient explodes?
             grad = torch.zeros_like(grad)
             
        mean_forward = x + (dt / 2) * grad
        proposed_x = mean_forward + self.step_size * noise
        
        # Backward step logic
        # We need gradient at proposed_x
        xp = proposed_x.detach().clone().requires_grad_(True)
        lp_p = log_prob_fn(xp)
        grad_p = torch.autograd.grad(lp_p, xp)[0]
        
        if torch.isnan(grad_p).any():
             # Fallback
             grad_p = torch.zeros_like(grad_p)

        mean_backward = xp + (dt / 2) * grad_p
        
        # q(x'|x) proportional to exp(-||x' - mean_fwd||^2 / (2 dt))
        # log q(x'|x) = -sum((x' - mean_fwd)**2) / (2 dt)
        
        log_q_fwd = -torch.sum((proposed_x - mean_forward)**2) / (2 * dt)
        log_q_bwd = -torch.sum((current_x - mean_backward)**2) / (2 * dt)
        
        return proposed_x, log_q_bwd - log_q_fwd

