import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

def generate_dataset_figure():
    print("Generating Dataset Visualization...")
    
    with h5py.File('datasets/photonic_data.h5', 'r') as f:
        # Load data
        spectra = f['data']['spectra'][:] # (N, W)
        params = f['data']['params'][:]   # (N, D)
        
        # Metadata access via json if needed, but we can assume W=200
        # Wavelengths 400-1000
        wavelengths = np.linspace(400, 1000, spectra.shape[1])
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Spectra Visualization
    # Plot first 50 spectra in thin grey lines
    for i in range(50):
        axes[0].plot(wavelengths, spectra[i], color='gray', alpha=0.1, lw=1)
        
    # Mean and Std
    mean_spec = np.mean(spectra, axis=0)
    std_spec = np.std(spectra, axis=0)
    
    axes[0].plot(wavelengths, mean_spec, 'b-', lw=2, label='Mean Spectrum')
    axes[0].fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec, color='blue', alpha=0.2, label='1 Std Dev')
    
    axes[0].set_title(f"Dataset Spectra (N={spectra.shape[0]})")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Thickness Distribution
    # Flatten params
    d_vals = params.flatten()
    
    axes[1].hist(d_vals, bins=30, color='teal', alpha=0.7, edgecolor='black')
    axes[1].set_title("Layer Thickness Distribution")
    axes[1].set_xlabel("Thickness (nm)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper/figures/dataset_stats.png', dpi=300)
    print("Saved paper/figures/dataset_stats.png")

if __name__ == "__main__":
    generate_dataset_figure()
