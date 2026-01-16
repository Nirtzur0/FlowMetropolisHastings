import torch
import numpy as np
import scipy.stats
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal

def test_gaussian_1d_ks():
    """
    Test if the sampler correctly samples from a 1D Standard Gaussian.
    Uses Kolmogorov-Smirnov test.
    """
    print("Running KS Test on 1D Gaussian...")
    
    # Target: N(0, 1)
    def log_prob(x):
        return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
        
    dim = 1
    
    # Setup Sampler with Global Proposal
    # Note: Even if Flow is untrained/random, proper MH should correct it to N(0,1).
    # This is the ultimate test of "Robustness" & "Exactness".
    
    flow = FlowProposal(dim, step_size=0.05, deterministic_trace=True, mixture_prob=0.2, broad_scale=2.0) # Mixture for robustness
    
    sampler = DiffusionMH(
        log_prob_fn=log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=1.0),
        global_proposal=flow,
        p_global=0.5 # Mixing
    )
    
    # Run a shorter chain for unit test speed
    n_samples = 1000
    samples, stats = sampler.run(torch.zeros(dim), n_samples, warmup=200, seed=999)
    samples_flat = samples.flatten()
    
    # KS Test
    # H0: samples come from N(0,1)
    # If p_value < 0.05, we reject H0 (Fail).
    # But for random tests, we should be careful. 
    # Let's require p > 0.001 to be "not obviously wrong".
    
    ks_stat, p_value = scipy.stats.kstest(samples_flat, 'norm')
    
    print(f"KS Stat: {ks_stat:.4f}, p-value: {p_value:.4f}")
    
    # Assert
    assert p_value > 0.001, f"KS Test failed! Samples do not look Gaussian. p={p_value}"
    
    # Also check mean/std
    mean = np.mean(samples_flat)
    std = np.std(samples_flat)
    print(f"Mean: {mean:.4f} (Expected 0)")
    print(f"Std: {std:.4f} (Expected 1)")
    
    assert abs(mean) < 0.1
    assert abs(std - 1.0) < 0.1

if __name__ == "__main__":
    test_gaussian_1d_ks()
