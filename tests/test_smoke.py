import torch
import numpy as np
import pytest
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel, MALAKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.targets.banana import BananaTarget

def test_sampler_runs_rwm():
    target = BananaTarget(dim=2)
    sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=2,
        local_kernel=RWMKernel(scale=0.1),
        p_global=0.0 # pure local
    )
    samples, stats = sampler.run(
        initial_x=torch.zeros(2),
        num_steps=50,
        warmup=5
    )
    assert samples.shape == (50, 2)
    assert stats['attempts_global'] == 0
    assert not np.isnan(samples).any()

def test_sampler_runs_flow_untrained():
    target = BananaTarget(dim=2)
    flow = FlowProposal(dim=2)
    sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=2,
        local_kernel=RWMKernel(scale=0.1),
        global_proposal=flow,
        p_global=0.5
    )
    samples, stats = sampler.run(
        initial_x=torch.zeros(2),
        num_steps=50,
        warmup=5
    )
    assert samples.shape == (50, 2)
    assert stats['attempts_global'] > 0
    assert not np.isnan(samples).any()
    
def test_density_consistency_shape():
    flow = FlowProposal(dim=2)
    x = torch.randn(10, 2)
    lp = flow.log_prob(x)
    assert lp.shape == (10,)
    assert not torch.isnan(lp).any()
