
import torch
import numpy as np
import pytest
from diffmcmc.targets.funnel import FunnelTarget

def test_funnel_shape():
    batch_size = 10
    dim = 5
    target = FunnelTarget(dim=dim)
    x = torch.randn(batch_size, dim)
    log_prob = target.log_prob(x)
    assert log_prob.shape == (batch_size,)
    
def test_funnel_values():
    # Manual calculation check for a known point
    # Case: y=0, others=0
    # log_p(y=0) = log(1/(3*sqrt(2pi)) * exp(0)) = -log(3) - 0.5*log(2pi)
    # log_p(others=0|y=0) => scale=exp(0)=1. others sum of D-1 terms.
    # each term: -0.5*log(2pi) - 0.5*0^2/1 = -0.5*log(2pi)
    # Total = -log(3) - 0.5*log(2pi) + (D-1)*(-0.5*log(2pi))
    #       = -log(3) - 0.5*D*log(2pi)
    
    dim = 4
    target = FunnelTarget(dim=dim)
    x = torch.zeros(1, dim)
    log_prob = target.log_prob(x)
    
    expected = -np.log(3) - 0.5 * dim * np.log(2 * np.pi)
    assert torch.isclose(log_prob, torch.tensor(expected, dtype=torch.float32), atol=1e-5)

def test_funnel_gradients():
    dim = 3
    target = FunnelTarget(dim=dim)
    x = torch.randn(1, dim, requires_grad=True)
    log_prob = target.log_prob(x)
    log_prob.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

if __name__ == "__main__":
    test_funnel_shape()
    test_funnel_values()
    test_funnel_gradients()
    print("All tests passed!")
