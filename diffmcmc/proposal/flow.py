import torch
import torch.nn as nn
import numpy as np
import warnings
from diffmcmc.proposal.nets import VelocityMLP, TimeResNet
from typing import Optional, Any, List

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
    def __init__(
        self,
        dim: int,
        model: Optional[nn.Module] = None,
        step_size: float = 0.1,
        deterministic_trace: bool = True,
        mixture_prob: float = 0.0,
        broad_scale: float = 2.0,
        use_resnet: bool = True,
        integrator: str = "rk4",
        enforce_divisible: bool = True,
        auto_adjust_step_size: bool = True,
        exact_discrete_logdet: bool = False,
        max_exact_trace_dim: int = 64,
        max_exact_logdet_dim: int = 32
    ):
        """
        Args:
            dim: Dimension of data.
            model: Velocity network.
            step_size: Integration step size.
            deterministic_trace: If True, uses hashed noise for density estimation.
            mixture_prob: Probability 'eta' of sampling from broad background distribution.
            broad_scale: Scale (std) of the broad background Gaussian.
            use_resnet: If True and model is None, uses TimeResNet. Otherwise VelocityMLP.
            integrator: Numerical integrator ("rk4" or "euler").
            enforce_divisible: Enforce step_size dividing 1.0 for consistent integration grids.
            auto_adjust_step_size: If True, adjust step_size to the nearest divisor of 1.0.
            exact_discrete_logdet: If True, compute discrete-step logdet via Jacobians (exact but expensive).
            max_exact_trace_dim: Max dimension for exact trace in log_prob_exact.
            max_exact_logdet_dim: Max dimension for exact discrete logdet computation.
        """
        super().__init__()
        self.dim = dim
        self.integrator = integrator
        self.exact_discrete_logdet = exact_discrete_logdet
        self.max_exact_trace_dim = max_exact_trace_dim
        self.max_exact_logdet_dim = max_exact_logdet_dim
        self.step_size, self.num_steps, self._step_size_divisible = self._resolve_step_size(
            step_size,
            enforce_divisible=enforce_divisible,
            auto_adjust=auto_adjust_step_size
        )
        self.deterministic_trace = deterministic_trace
        self.mixture_prob = mixture_prob
        self.broad_scale = broad_scale
        
        if model is None:
            if use_resnet:
                # Default configuration for TimeResNet
                self.net = TimeResNet(dim)
            else:
                self.net = VelocityMLP(dim)
        else:
            self.net = model
            
        # Fixed random vector for pseudo-hashing
        # This acts as a "salt" for the hash function
        self.register_buffer("hash_salt", torch.randn(dim))

    def _resolve_step_size(
        self,
        step_size: float,
        enforce_divisible: bool = True,
        auto_adjust: bool = True
    ) -> tuple[float, int, bool]:
        inv = 1.0 / step_size
        steps = int(round(inv))
        divisible = np.isclose(inv, steps, rtol=0.0, atol=1e-6)
        if not divisible:
            msg = (
                f"step_size={step_size} does not divide 1.0; "
                "numerical integration grid will be inconsistent."
            )
            if enforce_divisible and auto_adjust:
                new_step = 1.0 / steps
                warnings.warn(f"{msg} Adjusting step_size to {new_step}.", RuntimeWarning)
                return new_step, steps, True
            if enforce_divisible:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)
        return step_size, max(1, steps), divisible

    def set_step_size(self, step_size: float, enforce_divisible: bool = True, auto_adjust: bool = True) -> None:
        self.step_size, self.num_steps, self._step_size_divisible = self._resolve_step_size(
            step_size,
            enforce_divisible=enforce_divisible,
            auto_adjust=auto_adjust
        )

    def _rk4_step_func(self, f: Any, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Standard RK4 stepper for arbitrary function f(x, t)."""
        k1 = f(x, t)
        k2 = f(x + dt * 0.5 * k1, t + dt * 0.5)
        k3 = f(x + dt * 0.5 * k2, t + dt * 0.5)
        k4 = f(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _euler_step_func(self, f: Any, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Explicit Euler stepper."""
        return x + dt * f(x, t)

    def _step_func(self, f: Any) -> Any:
        if self.integrator == "rk4":
            return lambda x, t, dt: self._rk4_step_func(f, x, t, dt)
        if self.integrator == "euler":
            return lambda x, t, dt: self._euler_step_func(f, x, t, dt)
        raise ValueError(f"Unsupported integrator: {self.integrator}")

    def sample(self, num_samples: int) -> torch.Tensor:
        try:
            device = next(self.net.parameters()).device
        except StopIteration:
            # Fallback if model has no parameters (e.g. analytical flow)
            device = torch.device("cpu")
            
        # 1. Mixture Selection
        # Mask: 1 if broad, 0 if flow
        if self.mixture_prob > 0:
            is_broad = torch.rand(num_samples, device=device) < self.mixture_prob
            num_broad = is_broad.sum().item()
            num_flow = num_samples - num_broad
        else:
            is_broad = torch.zeros(num_samples, dtype=torch.bool, device=device)
            num_broad = 0
            num_flow = num_samples
            
        x_out = torch.zeros(num_samples, self.dim, device=device)
        
        # 2. Broad Samples
        if num_broad > 0:
            x_out[is_broad] = torch.randn(num_broad, self.dim, device=device) * self.broad_scale
            
        # 3. Flow Samples
        if num_flow > 0:
            z = torch.randn(num_flow, self.dim, device=device)
            # Integrate 0 -> 1 using RK4
            x = z
            t = 0.0
            steps = self.num_steps
            
            def dynamics(x_curr, t_curr):
                return self.net(x_curr, t_curr)

            step_fn = self._step_func(dynamics)
            with torch.no_grad():
                for _ in range(steps):
                    x = step_fn(x, t, self.step_size)
                    t += self.step_size
            
            # Place in output
            if self.mixture_prob > 0:
                x_out[~is_broad] = x
            else:
                x_out = x
                
        return x_out
        
    def _get_hutchinson_noise(self, x: torch.Tensor, probe_idx: int = 0) -> torch.Tensor:
        """
        Generate noise epsilon. 
        If deterministic_trace is True, use a fast pseudo-random hash of x.
        probe_idx allows generating multiple independent robust probes.
        """
        if not self.deterministic_trace:
            return torch.randn_like(x)
        
        # Fast Vectorized Pseudo-Hash
        # Map x -> (B, D) using a fixed random matrix
        if not hasattr(self, "proj_mat"):
             generator = torch.Generator(device=x.device).manual_seed(42)
             proj = torch.randn(self.dim, self.dim, device=x.device, generator=generator)
             self.register_buffer("proj_mat", proj)
             
        # y = x @ M
        y = torch.matmul(x, self.proj_mat) + self.hash_salt
        
        # Adjust 'seed' for different probes by adding offset to y
        # We add probe_idx * large_prime
        y = y + (probe_idx * 1337.0)
        
        # deterministic noise
        eps = torch.sin(y * 1000.0) 
        
        # Normalize to Unit Variance. Sin is variance 0.5.
        # Multiply by sqrt(2)
        return eps * 1.41421356

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Standard log prob using current settings (compatible with old interface)."""
        return self._log_prob_core(x, self.step_size, exact_trace=False, num_hutchinson=1)
        
    def log_prob_cheap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 1 evaluator:
        - Coarse step size (2x default)
        - Single Hutchinson probe
        """
        # Note: If dimensions are very small, cheap might just be exact. 
        # But let's follow the plan: coarse solve.
        coarse_step = self.step_size * 2.0
        log_q_flow = self._log_prob_core(x, coarse_step, exact_trace=False, num_hutchinson=1, is_cheap=True)
        return self._mix_density(x, log_q_flow)
        
    def log_prob_exact(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 evaluator:
        - Exact trace if D <= 64
        - High-fidelity mode if D > 64 (fine step, more probes)
        """
        fine_step = self.step_size # or even 0.5 * step_size? Let's use standard step size as "exact" reference.
        
        use_discrete_logdet = self.exact_discrete_logdet
        if use_discrete_logdet and self.dim > self.max_exact_logdet_dim:
            warnings.warn(
                "exact_discrete_logdet requested but dim exceeds max_exact_logdet_dim; "
                "falling back to trace-based logdet.",
                RuntimeWarning
            )
            use_discrete_logdet = False

        if self.dim <= self.max_exact_trace_dim:
            # Exact Trace
            log_q_flow = self._log_prob_core(
                x,
                fine_step,
                exact_trace=True,
                discrete_logdet=use_discrete_logdet
            )
        else:
            # High-fidelity approx
            log_q_flow = self._log_prob_core(
                x,
                fine_step,
                exact_trace=False,
                num_hutchinson=10,
                discrete_logdet=use_discrete_logdet
            )
            
        return self._mix_density(x, log_q_flow)
        
    def _mix_density(self, x: torch.Tensor, log_q_flow: torch.Tensor) -> torch.Tensor:
        """
        Compute log((1-eta)q_flow + eta * r_broad).
        """
        if self.mixture_prob <= 0.0:
            return log_q_flow
            
        # r_broad(x) = N(x | 0, broad_scale^2 I)
        # log r_broad = -0.5 * sum((x/scale)^2) - D * log(scale) - D/2 log(2pi)
        D = self.dim
        scale = self.broad_scale
        
        log_r_broad = -0.5 * torch.sum((x / scale)**2, dim=1) \
                      - D * np.log(scale) \
                      - 0.5 * D * np.log(2 * np.pi)
                      
        # logsumexp
        # log_p = log( (1-eta) exp(lqf) + eta exp(lrb) )
        #       = logaddexp( log(1-eta)+lqf, log(eta)+lrb )
        
        log_eta = np.log(self.mixture_prob)
        log_one_minus_eta = np.log(1.0 - self.mixture_prob)
        
        return torch.logaddexp(log_one_minus_eta + log_q_flow, log_eta + log_r_broad)

    def exactness_report(self) -> List[str]:
        warnings_list: List[str] = []
        if not self.deterministic_trace:
            warnings_list.append("deterministic_trace=False: log_q is stochastic; exactness requires pseudo-marginal treatment.")
        if not self._step_size_divisible:
            warnings_list.append("step_size does not divide 1.0; integration grid is inconsistent.")
        if self.dim > self.max_exact_trace_dim:
            warnings_list.append("log_prob_exact uses Hutchinson trace for dim > max_exact_trace_dim.")
        if not self.exact_discrete_logdet:
            warnings_list.append("log_prob_exact uses divergence integral, not discrete-step logdet.")
        if self.exact_discrete_logdet and self.dim > self.max_exact_logdet_dim:
            warnings_list.append("exact_discrete_logdet disabled due to max_exact_logdet_dim.")
        if self.integrator not in ("rk4", "euler"):
            warnings_list.append(f"Unsupported integrator '{self.integrator}'.")
        return warnings_list

    def is_exact(self) -> bool:
        return len(self.exactness_report()) == 0

    def log_prob_unbiased(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Pseudo-marginal mode requires an unbiased estimator of q(x), "
            "which FlowProposal does not provide."
        )

    def _log_prob_core(
        self,
        x: torch.Tensor,
        step_size: float,
        exact_trace: bool = False,
        num_hutchinson: int = 1,
        is_cheap: bool = False,
        discrete_logdet: bool = False
    ) -> torch.Tensor:
        """
        Compute log q(x) using Hutchinson trace estimator or exact trace with RK4 integration.
        Integration operates backwards from 1 -> 0.
        """
        # x has device info
        device = x.device
        batch_size = x.shape[0]
        
        xt = x.clone()
        zero = torch.zeros(batch_size, 1, device=device)
        log_jac_trace = zero.clone()
        
        # Time steps for backward integration (1 -> 0)
        step_size, steps, _ = self._resolve_step_size(
            step_size,
            enforce_divisible=False,
            auto_adjust=False
        )
        dt = -step_size
        t = 1.0
        
        # Prepare noise vectors if Hutchinson
        # If exact_trace is True, we don't need noise.
        # If num_hutchinson > 1, we need multiple probes.
        probes = []
        if not exact_trace and not discrete_logdet:
            for k in range(num_hutchinson):
                 probes.append(self._get_hutchinson_noise(xt, probe_idx=k))
        
        def dynamics(x_curr, t_curr):
            return self.net(x_curr, t_curr)

        step_fn = self._step_func(dynamics)

        if discrete_logdet:
            log_det = zero.clone()
            for _ in range(steps):
                def step_single(x_in):
                    return step_fn(x_in.unsqueeze(0), t, dt).squeeze(0)

                jac = torch.vmap(torch.func.jacrev(step_single))(xt)
                _, logabsdet = torch.linalg.slogdet(jac)
                log_det = log_det + logabsdet.unsqueeze(1)
                xt = step_fn(xt, t, dt)
                t += dt
        else:
            def dynamics_combined(state, t_curr):
                x_c, _ = state
                with torch.enable_grad():
                    x_in = x_c.detach().requires_grad_(True)
                    v = self.net(x_in, t_curr)
                    if exact_trace:
                        def dynamics_func(x_in_inner):
                            return self.net(x_in_inner.unsqueeze(0), t_curr).squeeze(0)
                        jac = torch.vmap(torch.func.jacrev(dynamics_func))(x_in)
                        trace_est = torch.vmap(torch.trace)(jac).unsqueeze(1)
                    else:
                        trace_est = torch.zeros(batch_size, 1, device=device)
                        for eps in probes:
                            def func_v(inputs):
                                return self.net(inputs, t_curr)
                            _, jvp_val = torch.autograd.functional.jvp(func_v, x_in, v=eps)
                            trace_sample = torch.sum(eps * jvp_val, dim=1, keepdim=True)
                            trace_est += trace_sample
                        trace_est = trace_est / num_hutchinson
                return v, trace_est

            for _ in range(steps):
                v1, tr1 = dynamics_combined((xt, log_jac_trace), t)
                v2, tr2 = dynamics_combined((xt + dt*0.5*v1, log_jac_trace + dt*0.5*tr1), t + dt*0.5)
                v3, tr3 = dynamics_combined((xt + dt*0.5*v2, log_jac_trace + dt*0.5*tr2), t + dt*0.5)
                v4, tr4 = dynamics_combined((xt + dt*v3, log_jac_trace + dt*tr3), t + dt)
                
                with torch.no_grad():
                    xt = xt + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
                    log_jac_trace = log_jac_trace + (dt / 6.0) * (tr1 + 2*tr2 + 2*tr3 + tr4)
                t += dt
            
        # Final z is xt (at t=0)
        log_prob_z = -0.5 * torch.sum(xt**2, dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi, device=device))

        if discrete_logdet:
            return log_prob_z + log_det.squeeze(1)
        return log_prob_z - log_jac_trace.squeeze(1)
