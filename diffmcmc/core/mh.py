import torch
import numpy as np
from tqdm import tqdm
from diffmcmc.core.kernels import AbstractKernel
from diffmcmc.diagnostics.metrics import compute_ess
from typing import Callable, Optional, Tuple, Dict, Any
import time

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
        stats = {
            "accept_local": 0, 
            "accept_global": 0, 
            "accept_global_stage1": 0,
            "attempts_local": 0, 
            "attempts_global": 0,
            "total_time_sec": 0.0,
            "ess_min": 0.0,
            "ess_per_sec": 0.0
        }
        
        # Caching for global moves
        # current_lq_cheap and current_lq_exact are valid only if current_x hasn't changed since last global computation,
        # OR if we update them.
        # Strategically: We compute them on demand.
        # But if we just did a global move, we have them.
        # If we did a local move, they are invalid.
        # Actually, for DA stage 2, we need log_q_exact(current_x).
        # We should cache it.
        
        cache = {
            "log_q_cheap": None,
            "log_q_exact": None
        }
        
        start_time = time.time()
        
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
                    if proposed_x.device != current_x.device:
                        proposed_x = proposed_x.to(current_x.device)

                    # --- STAGE 1: Cheap Evaluation ---
                    proposed_lq_cheap = self.global_proposal.log_prob_cheap(proposed_x.unsqueeze(0)).squeeze(0)
                    proposed_lp = self.log_prob_fn(proposed_x)
                    
                    # Ensure current_lq_cheap is available
                    if cache["log_q_cheap"] is None:
                        cache["log_q_cheap"] = self.global_proposal.log_prob_cheap(current_x.unsqueeze(0)).squeeze(0)
                    current_lq_cheap = cache["log_q_cheap"]
                        
                    # Stage 1 MH Ratio (using cheap q)
                    # log_alpha_1 = log_pi(x') - log_pi(x) + log_q_cheap(x) - log_q_cheap(x')
                    log_r1 = proposed_lp - current_lp + current_lq_cheap - proposed_lq_cheap
                    log_alpha_1 = min(0, log_r1.item())
                    
                    # Standard DA check
                    if torch.log(torch.rand(1).to(self.device)) < log_alpha_1:
                        stats["accept_global_stage1"] += 1
                        
                        # --- STAGE 2: Exact Evaluation ---
                        # Only compute expensive stuff now
                        proposed_lq_exact = self.global_proposal.log_prob_exact(proposed_x.unsqueeze(0)).squeeze(0)
                        
                        if cache["log_q_exact"] is None:
                            cache["log_q_exact"] = self.global_proposal.log_prob_exact(current_x.unsqueeze(0)).squeeze(0)
                        current_lq_exact = cache["log_q_exact"]
                        
                        # Stage 2 MH Ratio Correction
                        # log_alpha_2 = min(0, log_r_true) - log_alpha_1
                        # where log_r_true = log_pi(x') - log_pi(x) + log_q_exact(x) - log_q_exact(x')
                        
                        log_r_true = proposed_lp - current_lp + current_lq_exact - proposed_lq_exact
                        log_alpha_true = min(0, log_r_true.item())
                        
                        # Probability of acceptance at stage 2
                        # alpha_2 = alpha_true / alpha_1
                        # log_alpha_2 = log_alpha_true - log_alpha_1
                        
                        raw_log_alpha_2 = log_alpha_true - log_alpha_1
                        
                        # Numerical stability: enforce alpha_2 <= 1.
                        # Floating point errors can cause log_alpha_true to be slightly larger than log_alpha_1
                        # if the cheap approximation was very close but underestimated slightly.
                        log_alpha_2 = min(0.0, raw_log_alpha_2)
                             
                        if torch.log(torch.rand(1).to(self.device)) < log_alpha_2:
                            current_x = proposed_x
                            current_lp = proposed_lp
                            accepted = True
                            
                            # Update Cache
                            cache["log_q_cheap"] = proposed_lq_cheap
                            cache["log_q_exact"] = proposed_lq_exact
                            
                    else:
                        # Stage 1 rejected
                        pass

                except RuntimeError as e:
                     print(f"Global move warning: {e}")
                     accepted = False
                except Exception as e:
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
                    # If we moved locally, current_lq is now invalid for the NEW x.
                    # Invalidate cache.
                    cache["log_q_cheap"] = None
                    cache["log_q_exact"] = None
                    accepted = True
                    
                if accepted: stats["accept_local"] += 1
            
            if step >= warmup:
                chain.append(current_x.detach().cpu().numpy())
                
        total_time = time.time() - start_time
        stats["total_time_sec"] = total_time
        
        chain_arr = np.array(chain)
        if len(chain_arr) > 0:
            ess = compute_ess(chain_arr)
            min_ess = np.min(ess)
            stats["ess_min"] = float(min_ess)
            stats["ess_per_sec"] = float(min_ess / (total_time + 1e-9))
        
        return chain_arr, stats
