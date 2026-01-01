
import torch
import time
from diffmcmc.proposal.flow import FlowProposal

def benchmark_trace():
    print("Benchmarking Trace Calculation...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using device: {device}")
    
    dims = [2, 64, 128, 256]
    
    for D in dims:
        print(f"\n--- Dimension {D} ---")
        flow = FlowProposal(dim=D, deterministic_trace=False).to(device)
        x = torch.randn(16, D).to(device) # Batch size 16
        
        # Warmup
        _ = flow.log_prob_exact(x)
        
        # Measure
        start = time.time()
        for _ in range(10):
            _ = flow.log_prob_exact(x)
        end = time.time()
        
        avg_time = (end - start) / 10.0
        print(f"Avg time per log_prob_exact call (Batch 16): {avg_time:.4f} sec")
        print(f"Estimated time steps: {int(1.0/flow.step_size)}")
        print(f"Time per ODE step approx: {avg_time / int(1.0/flow.step_size):.4f} sec")

if __name__ == "__main__":
    benchmark_trace()
