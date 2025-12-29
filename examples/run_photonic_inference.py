import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.proposal.training import train_flow_matching
from diffmcmc.targets.thin_film import ThinFilmTarget
from diffmcmc.data.io import PhotonicHDF5Dataset

def run_experiment():
    print("--- Real-World Problem: Photonic Thin-Film Inference (HDF5) ---")
    
    # 1. Load Data Professionally
    dataset = PhotonicHDF5Dataset('datasets/photonic_data.h5')
    print(f"Loaded Dataset: {dataset.metadata.name}, N={len(dataset)}")
    
    # Pick a random test sample
    idx = 42
    sample = dataset[idx]
    obs_spectrum = sample['spectrum']
    true_params = sample['params'] # Thicknesses d
    true_log_params = torch.log(true_params)
    
    wavelengths = torch.from_numpy(dataset.metadata.wavelengths).float()
    
    # Material Pattern (Hardcoded matching the generator)
    # [Amb, L1, L2, L1, L2, L1, L2, L1, L2, L1, L2, Sub]
    n_ambient = 1.0
    n_1 = 1.45
    n_2 = 2.0
    n_sub = 3.5
    n_pattern = [n_ambient] + [n_1, n_2] * 5 + [n_sub]
    n_pattern = torch.tensor(n_pattern, dtype=torch.complex64)
    
    target = ThinFilmTarget(obs_spectrum, wavelengths, n_pattern, sigma=0.10)
    dim = 10 # 10 layers
    
    print(f"Target: Infer 10-layer thicknesses from {len(wavelengths)} spectral points.")
    
    # Smart Initialization: Start near truth for demo validation
    # In a real scenario without ground truth, you'd use more samples or MCMC tempering
    def find_starting_point_validated(true_log_params, target, dim):
        """
        For validation: perturb true params slightly and optimize back.
        This demonstrates the algorithm works when initialized reasonably.
        """
        print("Validated Initialization: Starting from perturbed true params...")
        
        # Perturb by 10% in log-space
        noise = torch.randn(dim) * 0.1
        perturbed = true_log_params + noise
        
        print(f"Initial LogProb (perturbed): {target.log_prob(perturbed.unsqueeze(0)).item():.2f}")
        
        # Gradient Optimization
        print("Refining with Adam Optimization...")
        x_opt = perturbed.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_opt], lr=0.02)
        
        for _ in range(200):
            optimizer.zero_grad()
            lp = target.log_prob(x_opt.unsqueeze(0))
            loss = -lp.sum()
            loss.backward()
            optimizer.step()
            
        print(f"Optimized Start: LogProb={-loss.item():.2f}")
        return x_opt.detach()

    # 1. Smart Init (using true params for validation)
    init_x = find_starting_point_validated(true_log_params, target, dim)
    
    # 2. RWMH Baseline
    print("Running RWMH...")
    
    start_time = time.time()
    rwm = DiffusionMH(
        target.log_prob,
        dim,
        RWMKernel(scale=0.01), 
        p_global=0.0
    )
    s_rwm, stats_rwm = rwm.run(init_x, 10000, warmup=2000, seed=42)
    acc_rwm = stats_rwm['accept_local'] / stats_rwm['attempts_local']
    print(f"RWMH Time: {time.time()-start_time:.2f}s, Accept: {acc_rwm:.2f}")
    
    # 3. DiffMCMC
    print("Running DiffMCMC...")
    
    # Train on RWM buffer (Pilot run)
    print("Training Flow...")
    # Use the end of the chain where it likely converged
    buffer_samples = s_rwm[2000::5].astype(np.float32)
    
    flow = FlowProposal(dim, deterministic_trace=True)
    train_flow_matching(flow, buffer_samples, epochs=150, batch_size=256, verbose=False)
    
    diff = DiffusionMH(
        target.log_prob,
        dim,
        RWMKernel(scale=0.02),
        global_proposal=flow,
        p_global=0.25
    )
    start_time = time.time()
    s_diff, stats_diff = diff.run(init_x, 10000, warmup=2000, seed=42)
    print(f"DiffMCMC Time: {time.time()-start_time:.2f}s")
    print(f"Global Accept Rate: {stats_diff['accept_global']/stats_diff['attempts_global']:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A. Spectrum Fit
    # Reconstruct spectrum from mean posterior
    mean_log_d = torch.mean(torch.from_numpy(s_diff).float(), dim=0)
    # Check posterior fit
    # Need to call solver manually
    with torch.no_grad():
        # Wrapper for reconstruction
        d_pred = torch.exp(mean_log_d).unsqueeze(0) # (1, 10)
        # Pad
        d_stack = torch.cat([torch.zeros(1,1), d_pred, torch.zeros(1,1)], dim=1)
        n_stack = n_pattern.unsqueeze(0)
        R_pred = target.solver(n_stack, d_stack, wavelengths.unsqueeze(0)).squeeze(0)
        
    axes[0, 0].plot(wavelengths, obs_spectrum, 'k-', lw=2, label='Observed (Truth)')
    axes[0, 0].plot(wavelengths, R_pred, 'r--', lw=2, label='DiffMCMC Mean')
    axes[0, 0].set_title("Spectral Reconstruction")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Reflectance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # B. Trace Plot (Layer 1 Thickness)
    axes[0, 1].plot(np.exp(s_rwm[:, 0]), alpha=0.6, label='RWMH')
    axes[0, 1].plot(np.exp(s_diff[:, 0]), alpha=0.6, color='orange', label='DiffMCMC')
    axes[0, 1].axhline(true_params[0].item(), color='k', ls='--', label='True')
    axes[0, 1].set_title("Trace: Layer 1 Thickness")
    axes[0, 1].set_ylabel("Thickness (nm)")
    axes[0, 1].legend()
    
    # C. Scatter: d1 vs d2
    axes[1, 0].scatter(np.exp(s_rwm[:, 0]), np.exp(s_rwm[:, 1]), s=5, alpha=0.3, label='RWMH')
    axes[1, 0].scatter(np.exp(s_diff[:, 0]), np.exp(s_diff[:, 1]), s=5, alpha=0.3, color='orange', label='DiffMCMC')
    axes[1, 0].plot(true_params[0], true_params[1], 'k*', ms=15, label='Truth')
    axes[1, 0].set_xlabel("d1 (nm)")
    axes[1, 0].set_ylabel("d2 (nm)")
    axes[1, 0].set_title("Posterior: Layer 1 vs 2")
    axes[1, 0].legend()
    
    # D. Marginal Error Boxplot
    # Relative error of mean vs truth
    mean_d_rwm = np.exp(np.mean(s_rwm, axis=0))
    mean_d_diff = np.exp(np.mean(s_diff, axis=0))
    truth = true_params.numpy()
    
    err_rwm = np.abs(mean_d_rwm - truth) / truth
    err_diff = np.abs(mean_d_diff - truth) / truth
    
    axes[1, 1].bar(np.arange(dim)-0.2, err_rwm, width=0.4, label='RWMH Error')
    axes[1, 1].bar(np.arange(dim)+0.2, err_diff, width=0.4, label='DiffMCMC Error')
    axes[1, 1].set_title("Relative Error per Layer")
    axes[1, 1].set_xlabel("Layer Index")
    axes[1, 1].set_ylabel("Rel Error")
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/photonic_results.png')
    print("Saved paper/figures/photonic_results.png")
    
    dataset.close()

if __name__ == "__main__":
    run_experiment()
