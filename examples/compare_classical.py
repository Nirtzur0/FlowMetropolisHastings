import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import differential_evolution

from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.proposal.training import train_flow_matching
from diffmcmc.targets.thin_film import ThinFilmTarget
from diffmcmc.data.io import PhotonicHDF5Dataset

def run_comparison():
    print("--- Benchmark: DiffMCMC vs Classical Differential Evolution (DE) ---")
    
    # Load Data
    dataset = PhotonicHDF5Dataset('datasets/photonic_data.h5')
    idx = 42
    sample = dataset[idx]
    obs_spectrum = sample['spectrum']
    true_params = sample['params'] # d_true
    wavelengths = torch.from_numpy(dataset.metadata.wavelengths).float()
    
    # Setup Target
    n_ambient = 1.0; n_1 = 1.45; n_2 = 2.0; n_sub = 3.5
    n_pattern = [n_ambient] + [n_1, n_2] * 5 + [n_sub]
    n_pattern = torch.tensor(n_pattern, dtype=torch.complex64)
    
    target = ThinFilmTarget(obs_spectrum, wavelengths, n_pattern, sigma=0.1)
    dim = 10
    
    # --- 1. Classical Differential Evolution ---
    print("\nRunning Classical DE (Scipy)...")
    
    def objective_func(x):
        # x is numpy array of params (log thickness or linear?)
        # Let's use linear bounds [50, 300] and convert to expected format
        # Target expects log-params if we keep consistency with DiffMCMC wrapper?
        # DiffMCMC wrapper expects Log-Params.
        # But DE bounds are nicer in Linear.
        # Let's optimize in Linear and Log inside objective.
        
        # Actually, let's optimize in LOG space to be fair to DiffMCMC
        # Bounds: log(50) ~ 3.9, log(300) ~ 5.7
        t_x = torch.tensor(x).float().unsqueeze(0)
        with torch.no_grad():
            lp = target.log_prob(t_x)
        return -lp.item()

    bounds = [(np.log(50.0), np.log(300.0))] * dim
    
    start_de = time.time()
    result_de = differential_evolution(
        objective_func, 
        bounds=bounds, 
        popsize=15, 
        maxiter=1000, 
        tol=0.01,
        polish=True
    )
    time_de = time.time() - start_de
    
    x_de_log = result_de.x
    x_de = np.exp(x_de_log)
    nll_de = result_de.fun
    
    print(f"DE Result: Time={time_de:.2f}s, NLL={nll_de:.2f}")
    
    # --- 2. DiffMCMC (With Gradient Init) ---
    print("\nRunning DiffMCMC (Gradient Init)...")
    
    # Gradient Init (Adam)
    # Find start
    def find_start(n_samples=2000):
        # Random search
        low = np.log(50.0); high = np.log(300.0)
        x_batch = torch.rand(n_samples, dim) * (high - low) + low
        # Batch eval
        log_probs = []
        with torch.no_grad():
            for i in range(0, n_samples, 500):
                 lp = target.log_prob(x_batch[i:i+500])
                 log_probs.append(lp)
        lp = torch.cat(log_probs)
        best_idx = torch.argmax(lp)
        best_x = x_batch[best_idx].clone().detach().requires_grad_(True)
        
        # Optimize
        optim = torch.optim.Adam([best_x], lr=0.05)
        for _ in range(200):
            optim.zero_grad()
            loss = -target.log_prob(best_x.unsqueeze(0)).sum()
            loss.backward()
            optim.step()
        return best_x.detach()
        
    start_diff = time.time()
    init_x = find_start()
    
    # Short RWMH
    rwm = DiffusionMH(target.log_prob, dim, RWMKernel(scale=0.01), p_global=0.0)
    s_rwm, _ = rwm.run(init_x, 5000, warmup=1000, seed=42)
    
    # DiffMCMC
    buffer = s_rwm[::5].astype(np.float32)
    flow = FlowProposal(dim, deterministic_trace=True)
    train_flow_matching(flow, buffer, epochs=50, verbose=False)
    
    diff = DiffusionMH(target.log_prob, dim, RWMKernel(scale=0.01), global_proposal=flow, p_global=0.2)
    s_diff, stats = diff.run(init_x, 5000, warmup=1000, seed=42)
    
    time_diff = time.time() - start_diff
    
    # Mean of DiffMCMC
    x_diff_log = np.mean(s_diff, axis=0)
    x_diff = np.exp(x_diff_log)
    
    # NLL of DiffMCMC Mean
    with torch.no_grad():
        nll_diff = -target.log_prob(torch.tensor(x_diff_log).float().unsqueeze(0)).item()
        
    print(f"DiffMCMC Result: Time={time_diff:.2f}s, NLL={nll_diff:.2f}")

    # --- 3. Analysis & Plotting ---
    
    # Metrics
    truth = true_params.numpy()
    
    mse_param_de = np.mean((x_de - truth)**2)
    mse_param_diff = np.mean((x_diff - truth)**2)
    
    print(f"\nParameter MSE: DE={mse_param_de:.2f}, DiffMCMC={mse_param_diff:.2f}")
    
    # Spectral Reconstruction
    def get_spectrum(x_log):
        d_val = torch.exp(torch.tensor(x_log)).unsqueeze(0)
        d_stack = torch.cat([torch.zeros(1,1), d_val, torch.zeros(1,1)], dim=1)
        n_stack = n_pattern.unsqueeze(0)
        with torch.no_grad():
            R = target.solver(n_stack, d_stack, wavelengths.unsqueeze(0)).squeeze(0)
        return R.numpy()
        
    R_de = get_spectrum(x_de_log)
    R_diff = get_spectrum(x_diff_log)
    R_true = obs_spectrum.numpy()
    
    mse_spec_de = np.mean((R_de - R_true)**2)
    mse_spec_diff = np.mean((R_diff - R_true)**2)
    
    print(f"Spectral MSE:   DE={mse_spec_de:.2e}, DiffMCMC={mse_spec_diff:.2e}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spectra
    axes[0].plot(wavelengths, R_true, 'k-', lw=2, label='Truth')
    axes[0].plot(wavelengths, R_de, 'b--', label=f'DE (MSE={mse_spec_de:.1e})')
    axes[0].plot(wavelengths, R_diff, 'r:', lw=2, label=f'DiffMCMC (MSE={mse_spec_diff:.1e})')
    axes[0].set_title("Spectral Fit Comparison")
    axes[0].legend()
    
    # Parameters Interleaved Bar Plot
    indices = np.arange(dim)
    width = 0.35
    axes[1].bar(indices - width/2, np.abs(x_de - truth)/truth, width, label='DE Error')
    axes[1].bar(indices + width/2, np.abs(x_diff - truth)/truth, width, label='DiffMCMC Error')
    axes[1].set_title("Parameter Relative Error")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("|Est - True| / True")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('paper/figures/classical_comparison.png')
    print("Saved paper/figures/classical_comparison.png")
    dataset.close()

if __name__ == "__main__":
    run_comparison()
