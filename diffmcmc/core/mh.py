import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from diffmcmc.core.kernels import AbstractKernel
from diffmcmc.core.acceptance import delayed_acceptance_log_r1, delayed_acceptance_log_r2
from diffmcmc.core.inference import InferenceMode, validate_inference_mode
from diffmcmc.diagnostics.metrics import compute_ess
from typing import Callable, Optional, Tuple, Dict, Any
import time

@dataclass
class SamplerState:
    x: torch.Tensor
    log_prob: torch.Tensor
    cache: Dict[str, Optional[torch.Tensor]]

class DiffusionMH:
    """
    Mixture-kernel Metropolis-Hastings sampler.
    Interleaves local moves (kernel) with global moves (flow proposal).
    """
    
    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        local_kernel: AbstractKernel,
        global_proposal: Optional[Any] = None,  # Placeholder for FlowProposal
        p_global: float = 0.2,
        device: str = "cpu",
        inference_mode: str = "approx",
        strict_exactness: bool = False
    ):
        self.log_prob_fn = log_prob_fn
        self.dim = dim
        self.local_kernel = local_kernel
        self.global_proposal = global_proposal
        self.p_global = p_global
        self.device = device
        self.inference_mode = inference_mode
        self.strict_exactness = strict_exactness
        self._inference_validated = False
        self.inference_report = None
        self._log_prob_cheap_fn = None
        self._log_prob_exact_fn = None
        
    def _validate_inference_mode(self) -> None:
        if self._inference_validated:
            return
        report = validate_inference_mode(self.inference_mode, self.global_proposal, strict=self.strict_exactness)
        self.inference_report = report
        self.inference_mode = report.effective_mode.value
        if self.global_proposal is not None:
            cheap_fn = getattr(self.global_proposal, "log_prob_cheap", None)
            exact_fn = getattr(self.global_proposal, "log_prob_exact", None)
            if report.effective_mode == InferenceMode.PSEUDO_MARGINAL:
                exact_fn = getattr(self.global_proposal, "log_prob_unbiased", None) or exact_fn
            if exact_fn is None:
                exact_fn = self.global_proposal.log_prob
            if cheap_fn is None:
                cheap_fn = exact_fn
            self._log_prob_cheap_fn = cheap_fn
            self._log_prob_exact_fn = exact_fn
        self._inference_validated = True

    def initialize_state(self, initial_x: torch.Tensor, seed: Optional[int] = None) -> SamplerState:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        current_x = initial_x.to(self.device)
        current_lp = self.log_prob_fn(current_x)
        cache = {
            "log_q_cheap": None,
            "log_q_exact": None
        }
        return SamplerState(x=current_x, log_prob=current_lp, cache=cache)

    def step(self, state: SamplerState, stats: Dict[str, int]) -> Tuple[SamplerState, Dict[str, Any]]:
        self._validate_inference_mode()
        info = {
            "is_global": False,
            "accepted": False,
            "stage1_accepted": False
        }
        is_global = False
        if self.global_proposal is not None and torch.rand(1).item() < self.p_global:
            is_global = True
        info["is_global"] = is_global

        current_x = state.x
        current_lp = state.log_prob
        cache = state.cache

        if is_global:
            stats["attempts_global"] += 1
            try:
                proposed_x = self.global_proposal.sample(1).squeeze(0)
                if proposed_x.device != current_x.device:
                    proposed_x = proposed_x.to(current_x.device)

                proposed_lq_cheap = self._log_prob_cheap_fn(proposed_x.unsqueeze(0)).squeeze(0)
                proposed_lp = self.log_prob_fn(proposed_x)

                if cache["log_q_cheap"] is None:
                    cache["log_q_cheap"] = self._log_prob_cheap_fn(current_x.unsqueeze(0)).squeeze(0)
                current_lq_cheap = cache["log_q_cheap"]

                log_r1 = delayed_acceptance_log_r1(
                    float(current_lp),
                    float(proposed_lp),
                    float(current_lq_cheap),
                    float(proposed_lq_cheap),
                )
                log_alpha_1 = min(0.0, log_r1)

                if torch.log(torch.rand(1, device=self.device)) < log_alpha_1:
                    stats["accept_global_stage1"] += 1
                    info["stage1_accepted"] = True

                    proposed_lq_exact = self._log_prob_exact_fn(proposed_x.unsqueeze(0)).squeeze(0)
                    if cache["log_q_exact"] is None:
                        cache["log_q_exact"] = self._log_prob_exact_fn(current_x.unsqueeze(0)).squeeze(0)
                    current_lq_exact = cache["log_q_exact"]

                    log_r2 = delayed_acceptance_log_r2(
                        float(current_lq_exact),
                        float(proposed_lq_exact),
                        float(current_lq_cheap),
                        float(proposed_lq_cheap),
                    )
                    log_alpha_2 = min(0.0, log_r2)

                    if torch.log(torch.rand(1, device=self.device)) < log_alpha_2:
                        state.x = proposed_x
                        state.log_prob = proposed_lp
                        cache["log_q_cheap"] = proposed_lq_cheap
                        cache["log_q_exact"] = proposed_lq_exact
                        info["accepted"] = True
            except RuntimeError as e:
                print(f"Global move warning: {e}")
            except Exception as e:
                print(f"Global move failed unexpectedly: {e}")

            if info["accepted"]:
                stats["accept_global"] += 1
        else:
            stats["attempts_local"] += 1
            proposed_x, log_q_ratio = self.local_kernel.propose(current_x, self.log_prob_fn)
            proposed_lp = self.log_prob_fn(proposed_x)
            log_alpha = proposed_lp - current_lp + log_q_ratio

            if torch.log(torch.rand(1, device=self.device)) < log_alpha:
                state.x = proposed_x
                state.log_prob = proposed_lp
                cache["log_q_cheap"] = None
                cache["log_q_exact"] = None
                info["accepted"] = True

            if info["accepted"]:
                stats["accept_local"] += 1

        return state, info
        
    def run(
        self,
        initial_x: torch.Tensor,
        num_steps: int,
        warmup: int = 0,
        seed: Optional[int] = None,
        progress: bool = True
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        self._validate_inference_mode()
        state = self.initialize_state(initial_x, seed=seed)

        chain = []
        stats = {
            "accept_local": 0,
            "accept_global": 0,
            "accept_global_stage1": 0,
            "attempts_local": 0,
            "attempts_global": 0,
            "total_time_sec": 0.0,
            "ess_min": 0.0,
            "ess_per_sec": 0.0
        }

        start_time = time.time()

        iterator = range(num_steps + warmup)
        if progress:
            iterator = tqdm(iterator, desc="Sampling")
        for step in iterator:
            state, _ = self.step(state, stats)
            if step >= warmup:
                chain.append(state.x.detach().cpu().numpy())

        total_time = time.time() - start_time
        stats["total_time_sec"] = total_time

        chain_arr = np.array(chain)
        if len(chain_arr) > 0:
            ess = compute_ess(chain_arr)
            min_ess = np.min(ess)
            stats["ess_min"] = float(min_ess)
            stats["ess_per_sec"] = float(min_ess / (total_time + 1e-9))

        return chain_arr, stats
