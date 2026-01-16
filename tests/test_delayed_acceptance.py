import torch
import numpy as np
import pytest
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.targets.mog import GaussianMixtureTarget

def test_delayed_acceptance_mog():
    """
    Test that the Delayed Acceptance Global Move runs correctly on a simple 2D MoG target.
    We check for stage 1 and stage 2 metrics and basic chain validity.
    """
    torch.manual_seed(42)
    
    # 2D MoG with 2 modes
    target = GaussianMixtureTarget(dim=2, centers=[[-3.0, -3.0], [3.0, 3.0]], scale=1.0)
    
    # Flow Proposal
    # Untrained flow (random Init) acts as a complex proposal.
    # Mixture prob ensured broad coverage.
    flow = FlowProposal(dim=2, step_size=0.1, deterministic_trace=True, mixture_prob=0.1, broad_scale=3.0)
    
    # MH Sampler
    # Use RWM for local moves
    kernel = RWMKernel(scale=0.5)
    
    # High p_global to stress test global moves
    sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=2,
        local_kernel=kernel,
        global_proposal=flow,
        p_global=0.4, 
        device="cpu"
    )
    
    # Run
    # Start at origin
    x0 = torch.zeros(2)
    chain, stats = sampler.run(x0, num_steps=100, warmup=0, seed=123)
    
    print("\nSampler Stats:", stats)
    
    # Basic assertions
    assert chain.shape == (100, 2)
    assert stats["attempts_global"] > 0
    assert "accept_global_stage1" in stats
    assert "ess_per_sec" in stats
    assert stats["total_time_sec"] > 0
    
    # Check consistency
    # accept_global <= accept_global_stage1
    assert stats["accept_global"] <= stats["accept_global_stage1"]
    
    # Check that caching works implicitly? 
    # Hard to check internal state from here without mocking.
    # But if it crashed, we'd know.

if __name__ == "__main__":
    test_delayed_acceptance_mog()
