import torch
import numpy as np
from diffmcmc.proposal.flow import FlowProposal

def test_flow_reversibility():
    """
    Check if integrating forward and then backward recovers the original point.
    """
    dim = 2
    flow = FlowProposal(dim)
    
    z = torch.randn(10, dim)
    
    # Forward: 0 -> 1
    x = z
    t = 0.0
    dt = 0.1
    steps = int(1.0/dt)
    
    def dynamics(x_curr, t_curr):
        return flow.net(x_curr, t_curr)

    for _ in range(steps):
        x = flow._rk4_step_func(dynamics, x, t, dt)
        t += dt
        
    # Backward: 1 -> 0
    z_rec = x
    t = 1.0
    dt = -0.1
    
    for _ in range(steps):
        z_rec = flow._rk4_step_func(dynamics, z_rec, t, dt)
        t += dt
        
    # Check error
    # Note: RK4 with large steps on random init net might have some error, but should be small.
    err = torch.mean((z - z_rec)**2)
    # print(f"Reconstruction MSE: {err.item()}")
    assert err < 1e-4

def test_density_values_vs_gradient():
    """
    Check if log_prob output is somewhat consistent with brute force log_det using autograd?
    Or just check if it runs. Exact trace match is hard due to stochastic trace estimation.
    For this test we can mock the noise to be deterministic? 
    (Not easy without changing code).
    
    Instead, just check that it produces finite values and runs.
    """
    flow = FlowProposal(dim=2)
    x = torch.randn(5, 2)
    lp = flow.log_prob(x)
    assert torch.isfinite(lp).all()
