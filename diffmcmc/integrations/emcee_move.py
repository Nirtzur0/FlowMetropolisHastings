import numpy as np
import torch
from emcee.moves import Move
from diffmcmc.core.inference import InferenceMode, validate_inference_mode

class DiffusionMove(Move):
    """
    An emcee Move that uses a trained Global Proposal (Flow).
    """
    def __init__(self, flow_proposal, p_global=0.2, local_move=None, inference_mode: str = "approx"):
        self.flow = flow_proposal
        self.p_global = p_global
        self.local_move = local_move # Fallback local move
        report = validate_inference_mode(inference_mode, self.flow, strict=False)
        self.inference_mode = report.effective_mode
        if self.inference_mode == InferenceMode.EXACT:
            self._log_prob = getattr(self.flow, "log_prob_exact", self.flow.log_prob)
        else:
            self._log_prob = self.flow.log_prob
        
    def propose(self, model, state):
        """
        model: emcee model (has log_prob)
        state: emcee State
        """
        # emcee passes a batch of walkers.
        # state.coords: (n_walkers, dim)
        n_walkers, dim = state.coords.shape
        device = next(self.flow.net.parameters()).device
        
        # We process all walkers.
        # For each walker, we decide global vs local? 
        # Usually emcee moves apply to all walkers or subsets. 
        # Let's apply global to ALL walkers with probability p_global (per step check? NO, Move is called).
        # Emcee calls `move.propose`.
        # We can just return the result of global proposal.
        # But we want a mixture.
        # Emcee doesn't natively mix moves inside one class unless we do it.
        # Better: User puts this in the list of moves passed to Sampler with weights.
        # So this class should ONLY do the global move. The mixing is handled by emcee.
        
        # Global Move Proposal
        # x' ~ q(x')
        x_curr = torch.tensor(state.coords).float().to(device)
        
        # Sample new states
        try:
            x_prop = self.flow.sample(n_walkers).cpu().numpy() # (N, D)
            x_prop_t = torch.tensor(x_prop).float().to(device)
            
            # Compute q(x) and q(x')
            # q(x) - density of CURRENT states
            log_q_curr = self._log_prob(x_curr).cpu().numpy()
            
            # q(x') - density of PROPOSED states
            log_q_prop = self._log_prob(x_prop_t).cpu().numpy()
            
            # MH factor = q(x)/q(x')
            # log_factor = log_q_curr - log_q_prop
            log_factor = log_q_curr - log_q_prop
            
            return x_prop, log_factor
            
        except Exception as e:
            print(f"Flow sampling failed: {e}")
            # Return no-op
            return state.coords, np.zeros(n_walkers)
