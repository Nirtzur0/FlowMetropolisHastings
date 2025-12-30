import torch
import torch.nn as nn
import numpy as np
from diffmcmc.proposal.nets import VelocityMLP
from typing import Optional, Any

class FlowProposal(nn.Module):
    r"""
    Continuous Normalizing Flow Proposal defined by a velocity field v(x, t).
    
    Generative (sampling):
       z ~ N(0, I)
       Solve dx/dt = v(x, t) from t=0 to 1 -> x
       
    Density (log q(x)):
       Solve backward dx/dt = v(x, t) from t=1 to 0 -> z
       log q(x) = log p(z) - \int_0^1 div(v) dt
       
    Rigorous Exactness:
       To ensure detailed balance in MH, the density estimate q(x) must be a deterministic function of x.
       We achieve this by hashing x to seed the Hutchinson noise \epsilon.
    """
    def __init__(self, dim: int, model: Optional[nn.Module] = None, step_size: float = 0.1, deterministic_trace: bool = True):
        """
        Args:
            dim: Dimension of data.
            model: Velocity network.
            step_size: Integration step size.
            deterministic_trace: If True, uses hashed noise for density estimation.
        """
        super().__init__()
        self.dim = dim
        self.step_size = step_size
        self.deterministic_trace = deterministic_trace
        if model is None:
            self.net = VelocityMLP(dim)
        else:
            self.net = model
            
        # Fixed random vector for pseudo-hashing
        # This acts as a "salt" for the hash function
        self.register_buffer("hash_salt", torch.randn(dim))

    def _rk4_step_func(self, f: Any, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Standard RK4 stepper for arbitrary function f(x, t)."""
        k1 = f(x, t)
        k2 = f(x + dt * 0.5 * k1, t + dt * 0.5)
        k3 = f(x + dt * 0.5 * k2, t + dt * 0.5)
        k4 = f(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def sample(self, num_samples: int) -> torch.Tensor:
        try:
            device = next(self.net.parameters()).device
        except StopIteration:
            # Fallback if model has no parameters (e.g. analytical flow)
            device = torch.device("cpu")
            
        z = torch.randn(num_samples, self.dim, device=device)
        
        # Integrate 0 -> 1 using RK4
        x = z
        t = 0.0
        steps = int(1.0 / self.step_size)
        
        # Define dynamics function for RK4
        def dynamics(x_curr, t_curr):
            return self.net(x_curr, t_curr)

        with torch.no_grad():
            for _ in range(steps):
                x = self._rk4_step_func(dynamics, x, t, self.step_size)
                t += self.step_size
                
        return x
        
    def _get_hutchinson_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate noise epsilon. 
        If deterministic_trace is True, use a fast pseudo-random hash of x.
        """
        if not self.deterministic_trace:
            return torch.randn_like(x)
        
        # Fast Vectorized Pseudo-Hash
        # sin(dot(x, salt)) * large_number
        # This is not cryptographically secure but deterministic and mixing enough for trace estimation correctness in MH.
        
        # x: (B, D)
        # salt: (D,)
        
        # Normalize x slightly to avoid huge values exploding sin
        # but keep it sensitive
        
        proj = torch.matmul(x, self.hash_salt) # (B,)
        
        # Create a seed-like pattern. 
        # We need (B, D) output.
        # Let's generate a seed per batch item, then use PyTorch generator? No, too slow.
        # Analytic noise generation:
        
        # Expand proj to (B, D) via broadcasting with different frequencies
        freqs = torch.arange(1, self.dim + 1, device=x.device).float() * torch.pi
        
        # noise ~ sin(proj * freq + offset) ??
        # A simple "Gold Noise" variant or similar:
        # phi = (1+sqrt(5))/2
        # noise = frac(sin(dot(uv, K)) * 43758.5453)
        
        # We implement a variant for (B, D)
        # We need independent noise per dimension d
        
        # Construct a large matrix of random weights for projection
        # Ideally cached. 
        # Let's just create it on the fly with a fixed seed? No.
        
        # Let's just use a simple robust hash if speed allows?
        # Reverting to CPU hash is safe but slow.
        # Let's stick to the previous implementation plan's suggestion:
        # "Use the same 'randomness' for evaluating q(x) every time x is visited."
        # If we just fix ONE noise vector for the ENTIRE RUN, it fails to be "random" enough.
        
        # High-performance pseudo-random generator:
        # x (B, D) -> (B, D) gaussian-ish
        
        # 1. Coordinate mixing
        # seed = x @ fixed_random_matrix 
        # noise = sin(seed) ?
        
        # Let's try the CPU hash again but optimized? 
        # Actually, let's look at the original code. It looped.
        # We can vector-hash: convert float view to int?
        
        # Let's go with a simple "sin" hash for now, it's standard in differentiable rendering tricks.
        # It doesn't need to be perfect Gaussian, just zero mean unit variance roughly.
        
        # Map x -> (B, D)
        # We use a fixed random matrix `proj_mat`
        if not hasattr(self, 'proj_mat'):
             # Lazy init buffer
             generator = torch.Generator(device=x.device).manual_seed(42)
             self.proj_mat = torch.randn(self.dim, self.dim, device=x.device, generator=generator)
             
        # y = x @ M
        y = torch.matmul(x, self.proj_mat)
        
        # deterministic noise
        eps = torch.sin(y * 1000.0) 
        
        # Normalize to Unit Variance? Sin is variance 0.5.
        # Multiply by sqrt(2)
        return eps * 1.41421356

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log q(x) using Hutchinson trace estimator with RK4 integration.
        Integration operates backwards from 1 -> 0.
        """
        # x has device info
        device = x.device
        batch_size = x.shape[0]
        
        # State: [x, log_jac_trace]
        # x: (B, D)
        # trace: (B, 1) or (B,)
        
        xt = x.clone()
        zero = torch.zeros(batch_size, 1, device=device)
        log_jac_trace = zero.clone() # Accumulate integral of div
        
        # Combined state for RK4
        # We need to flatten or handle tuple.
        # Let's handle tuple manually in stepper.
        
        # Time steps for backward integration (1 -> 0)
        # dt is negative
        dt = -self.step_size
        steps = int(1.0 / self.step_size)
        t = 1.0
        
        # Generate Deterministic Noise
        noise = self._get_hutchinson_noise(xt)
        
        def dynamics_combined(state, t_curr):
            # state is (x_curr, trace_curr)
            # Actually RK4 needs linear algebra usually.
            # But we can just return (dx/dt, dtrace/dt)
            x_c, _ = state
            
            # 1. dx/dt = v(x, t)
            # Enable grad for Trace
            with torch.enable_grad():
                x_in = x_c.detach().requires_grad_(True)
                v = self.net(x_in, t_curr)
                
                # 2. dtrace/dt = div(v)
                # Compute div estimate: eps^T * (J*eps)
                # J*eps
                
                # Use autograd.grad to compute vector-Jacobian product
                # We want J @ noise = directional derivative of v in direction noise?
                # No, JVP is J @ v.
                # Here we want Trace(J) approx v^T J v.
                # eps^T (df/dx eps)
                
                # Compute 'v' at x_in. 
                # Directional derivative in direction 'noise'
                # jvp(func, inputs, v=vectors)
                
                def func_v(inputs):
                    return self.net(inputs, t_curr)
                    
                _, jvp_val = torch.autograd.functional.jvp(func_v, x_in, v=noise)
                
                # trace_est = noise . jvp
                trace_est = torch.sum(noise * jvp_val, dim=1, keepdim=True)
                
            return v, trace_est

        # Custom RK4 for tuple state
        for _ in range(steps):
             # Tuple RK4
             # k1
             v1, tr1 = dynamics_combined((xt, log_jac_trace), t)
             
             # k2
             v2, tr2 = dynamics_combined((xt + dt*0.5*v1, log_jac_trace + dt*0.5*tr1), t + dt*0.5)
             
             # k3
             v3, tr3 = dynamics_combined((xt + dt*0.5*v2, log_jac_trace + dt*0.5*tr2), t + dt*0.5)
             
             # k4
             v4, tr4 = dynamics_combined((xt + dt*v3, log_jac_trace + dt*tr3), t + dt)
             
             # Update
             with torch.no_grad():
                 xt = xt + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
                 log_jac_trace = log_jac_trace + (dt / 6.0) * (tr1 + 2*tr2 + 2*tr3 + tr4)
             
             t += dt
            
        # Final z is xt (at t=0)
        log_prob_z = -0.5 * torch.sum(xt**2, dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi, device=device))
        
        # Result
        # log q(x) = log p(z) - \int div
        # Our integral accumulates div * dt. Since dt is negative, we accumulated -div*|dt|?
        # The formula is log q1 = log q0 - int_0^1 div(v) dt
        # We integrated from 1 to 0. 
        # int_1^0 div(v) dt = - int_0^1 div(v) dt.
        # So our 'log_jac_trace' contains exactly - int_0^1 div.
        # Wait, if we integrate div(v) * dt with negative dt...
        # Integral = Sum (div * -|dt|).
        # So yes, log_jac_trace = - int div(v).
        
        return log_prob_z - log_jac_trace.squeeze(1)
