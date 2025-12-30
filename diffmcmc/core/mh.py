import torch
import numpy as np
from tqdm import tqdm
from diffmcmc.core.kernels import AbstractKernel
from typing import Callable, Optional, Tuple, Dict, Any

class DiffusionMH:
    """
    Mixture-kernel Metropolis-Hastings sampler.
    Interleaves local moves (kernel) with global moves (flow proposal).
    """
    
    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        local_kernel: AbstractKernel,
        global_proposal: Optional[Any] = None,  # Placeholder for FlowProposal
        p_global: float = 0.2,
        device: str = "cpu"
    ):
        self.log_prob_fn = log_prob_fn
        self.dim = dim
        self.local_kernel = local_kernel
        self.global_proposal = global_proposal
        self.p_global = p_global
        self.device = device
        
    def run(self, initial_x: torch.Tensor, num_steps: int, warmup: int = 0, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        current_x = initial_x.to(self.device)
        current_lp = self.log_prob_fn(current_x)
        
        # If we have a global proposal, we might want to cache log_q(x)
        current_lq = None
        if self.global_proposal is not None:
             # Initial global density check (if we started from random)
             current_lq = self.global_proposal.log_prob(current_x.unsqueeze(0)).squeeze(0)

        chain = []
        stats = {"accept_local": 0, "accept_global": 0, "attempts_local": 0, "attempts_global": 0}
        
        iterator = tqdm(range(num_steps + warmup), desc="Sampling")
        for step in iterator:
            
            # 1. Decide Move Type
            is_global = False
            if self.global_proposal is not None and torch.rand(1).item() < self.p_global:
                is_global = True
                
            accepted = False
            
            if is_global:
                stats["attempts_global"] += 1
                # --- GLOBAL MOVE ---
                # q(x') independent of x
                try:
                    proposed_x = self.global_proposal.sample(1).squeeze(0) # (D,)
                    # Depending on implementation, sample might return on same device or CPU. Ensure device match.
                    if proposed_x.device != current_x.device:
                        proposed_x = proposed_x.to(current_x.device)

                    proposed_lq = self.global_proposal.log_prob(proposed_x.unsqueeze(0)).squeeze(0)
                    proposed_lp = self.log_prob_fn(proposed_x)
                    
                    # MH Ratio:
                    # alpha = min(1, (pi(x') * q(x)) / (pi(x) * q(x')))
                    # log_alpha = log_pi(x') + log_q(x) - log_pi(x) - log_q(x')
                    
                    # NOTE: current_lq needs to be valid. If it's None (first step or model changed), compute it.
                    if current_lq is None:
                        current_lq = self.global_proposal.log_prob(current_x.unsqueeze(0)).squeeze(0)
                        
                    log_alpha = proposed_lp + current_lq - current_lp - proposed_lq
                    
                    if torch.log(torch.rand(1).to(self.device)) < log_alpha:
                        current_x = proposed_x
                        current_lp = proposed_lp
                        current_lq = proposed_lq
                        accepted = True
                except RuntimeError as e:
                     # Catch specific runtime errors (e.g. numerical stability in flow)
                    print(f"Global move warning: {e}")
                    accepted = False
                except Exception as e:
                    # For other unexpected errors, we should likely raise or log seriously.
                    # Retaining print for now but making it clearer it's a critical failure in proposal.
                    print(f"Global move failed unexpectedly: {e}")
                    accepted = False

                if accepted: stats["accept_global"] += 1
                
            else:
                stats["attempts_local"] += 1
                # --- LOCAL MOVE ---
                proposed_x, log_q_ratio = self.local_kernel.propose(current_x, self.log_prob_fn)
                proposed_lp = self.log_prob_fn(proposed_x)
                
                # MH Ratio:
                # alpha = min(1, (pi(x') * K(x|x')) / (pi(x) * K(x'|x)))
                # log_alpha = log_pi(x') - log_pi(x) + log_q_ratio
                log_alpha = proposed_lp - current_lp + log_q_ratio
                
                if torch.log(torch.rand(1).to(self.device)) < log_alpha:
                    current_x = proposed_x
                    current_lp = proposed_lp
                    # If we moved locally, current_lq is now invalid for the NEW x
                    # We can lazily recompute it only when the next global move happens, 
                    # OR compute it now. Lazy is better for performance if p_global is low.
                    current_lq = None 
                    accepted = True
                    
                if accepted: stats["accept_local"] += 1
            
            if step >= warmup:
                chain.append(current_x.detach().cpu().numpy())
                
        return np.array(chain), stats
