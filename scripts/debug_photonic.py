"""
Diagnostic script to debug why photonic inference is failing.
Tests:
1. TMM solver correctness: Does true_params produce obs_spectrum?
2. Likelihood sanity: What is log_prob(true_params)?
3. Gradient flow: Can we optimize FROM true params and stay there?
"""
import torch
import numpy as np

from diffmcmc.targets.thin_film import ThinFilmTarget
from diffmcmc.data.io import PhotonicHDF5Dataset

def run_diagnostics():
    print("=" * 60)
    print("PHOTONIC INFERENCE DIAGNOSTICS")
    print("=" * 60)
    
    # Load data
    dataset = PhotonicHDF5Dataset('datasets/photonic_data.h5')
    idx = 42
    sample = dataset[idx]
    obs_spectrum = sample['spectrum']
    true_params = sample['params']  # These are LINEAR thicknesses
    
    print(f"\n1. TRUE PARAMETERS (Linear thickness in nm):")
    print(f"   {true_params.numpy()}")
    print(f"   Min: {true_params.min():.1f}, Max: {true_params.max():.1f}")
    
    # Setup target
    wavelengths = torch.from_numpy(dataset.metadata.wavelengths).float()
    n_ambient = 1.0; n_1 = 1.45; n_2 = 2.0; n_sub = 3.5
    n_pattern = [n_ambient] + [n_1, n_2] * 5 + [n_sub]
    n_pattern_tensor = torch.tensor(n_pattern, dtype=torch.complex64)
    
    target = ThinFilmTarget(obs_spectrum, wavelengths, n_pattern_tensor, sigma=0.1)
    
    # Test 1: What is the log_prob of TRUE params?
    true_log_params = torch.log(true_params).unsqueeze(0)  # Target expects log-space
    with torch.no_grad():
        lp_true = target.log_prob(true_log_params)
    print(f"\n2. LOG_PROB OF TRUE PARAMS: {lp_true.item():.2f}")
    
    # Test 2: Reconstruct spectrum from true params
    with torch.no_grad():
        d_true = true_params.unsqueeze(0)  # (1, 10)
        d_stack = torch.cat([torch.zeros(1,1), d_true, torch.zeros(1,1)], dim=1)
        n_stack = n_pattern_tensor.unsqueeze(0)
        R_reconstructed = target.solver(n_stack, d_stack, wavelengths.unsqueeze(0)).squeeze(0)
    
    mse_reconstruction = torch.mean((R_reconstructed - obs_spectrum)**2).item()
    print(f"\n3. SPECTRUM RECONSTRUCTION MSE (True params): {mse_reconstruction:.6f}")
    print(f"   Expected MSE (noise^2): {0.1**2:.4f} = 0.01")
    
    if mse_reconstruction > 0.05:
        print("   ⚠️  WARNING: Reconstruction MSE is HIGH - TMM solver may have issues!")
    else:
        print("   ✓ Reconstruction looks reasonable")
    
    # Test 3: Random search - what's the best we find?
    print(f"\n4. RANDOM SEARCH (5000 samples):")
    low = np.log(50.0)
    high = np.log(300.0)
    x_batch = torch.rand(5000, 10) * (high - low) + low
    
    with torch.no_grad():
        log_probs = []
        for i in range(0, 5000, 500):
            lp = target.log_prob(x_batch[i:i+500])
            log_probs.append(lp)
        log_probs = torch.cat(log_probs)
    
    best_idx = torch.argmax(log_probs)
    best_lp = log_probs[best_idx].item()
    best_x = x_batch[best_idx]
    
    print(f"   Best random LogProb: {best_lp:.2f}")
    print(f"   True params LogProb: {lp_true.item():.2f}")
    print(f"   Best random params (exp): {torch.exp(best_x).numpy()}")
    
    if best_lp > lp_true.item():
        print("   ⚠️  WARNING: Random sample has HIGHER LogProb than true params!")
        print("   This means there's a BETTER solution than truth - DEGENERACY confirmed")
    
    # Test 4: Optimize from TRUE params - do we stay?
    print(f"\n5. GRADIENT OPTIMIZATION FROM TRUE PARAMS:")
    x_opt = true_log_params.clone().squeeze(0).requires_grad_(True)
    optimizer = torch.optim.Adam([x_opt], lr=0.01)
    
    initial_lp = target.log_prob(x_opt.unsqueeze(0)).item()
    print(f"   Initial LogProb: {initial_lp:.2f}")
    
    for _ in range(100):
        optimizer.zero_grad()
        loss = -target.log_prob(x_opt.unsqueeze(0)).sum()
        loss.backward()
        optimizer.step()
    
    final_lp = -loss.item()
    print(f"   Final LogProb: {final_lp:.2f}")
    print(f"   Final params (exp): {torch.exp(x_opt.detach()).numpy()}")
    
    param_drift = torch.abs(torch.exp(x_opt.detach()) - true_params).mean().item()
    print(f"   Mean param drift from truth: {param_drift:.2f} nm")
    
    if param_drift > 10:
        print("   ⚠️  WARNING: Optimizer drifted away from true params!")
    
    # Test 5: Compare spectra
    print(f"\n6. SPECTRUM COMPARISON:")
    print(f"   Observed spectrum range: [{obs_spectrum.min():.3f}, {obs_spectrum.max():.3f}]")
    print(f"   Reconstructed (true) range: [{R_reconstructed.min():.3f}, {R_reconstructed.max():.3f}]")
    
    dataset.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_diagnostics()
