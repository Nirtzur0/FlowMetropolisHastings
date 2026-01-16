import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diffmcmc.core.mh import DiffusionMH
from diffmcmc.diagnostics.metrics import compute_ess, compute_rhat_rank_normalized
from diffmcmc.proposal.training import train_flow_matching

@dataclass
class AdaptConfig:
    enabled: bool = True
    interval: int = 50
    target_accept_local: float = 0.234
    target_accept_global: float = 0.3
    adapt_rate: float = 0.05
    min_local_step: float = 1e-4
    max_local_step: float = 10.0
    min_p_global: float = 0.01
    max_p_global: float = 0.8

@dataclass
class TrainingSchedule:
    enabled: bool = False
    train_every: int = 200
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    max_updates: int = 5
    buffer_size: int = 5000

@dataclass
class OrchestrationResult:
    chains: np.ndarray
    stats: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]

class SamplerOrchestrator:
    """
    Multi-chain orchestration with warmup-only adaptation and training schedules.
    """
    def __init__(
        self,
        sampler: Optional[DiffusionMH] = None,
        sampler_factory: Optional[Callable[[], DiffusionMH]] = None,
        adapt_config: Optional[AdaptConfig] = None,
        training_schedule: Optional[TrainingSchedule] = None,
        share_global_proposal: bool = False
    ):
        if sampler is None and sampler_factory is None:
            raise ValueError("Provide sampler or sampler_factory.")
        self.base_sampler = sampler
        self.sampler_factory = sampler_factory
        self.adapt_config = adapt_config or AdaptConfig()
        self.training_schedule = training_schedule or TrainingSchedule()
        self.share_global_proposal = share_global_proposal

    def _clone_sampler(self) -> DiffusionMH:
        if self.sampler_factory is not None:
            return self.sampler_factory()
        sampler = copy.deepcopy(self.base_sampler)
        if self.share_global_proposal and self.base_sampler is not None:
            sampler.global_proposal = self.base_sampler.global_proposal
        return sampler

    def run_chains(
        self,
        initial_xs: Sequence[torch.Tensor],
        num_steps: int,
        warmup: int = 0,
        seeds: Optional[Sequence[int]] = None,
        progress: bool = True,
        compute_diagnostics: bool = True
    ) -> OrchestrationResult:
        num_chains = len(initial_xs)
        if seeds is None:
            seeds = [None] * num_chains
        if len(seeds) != num_chains:
            raise ValueError("seeds length must match number of chains.")

        chains = []
        stats_list = []

        for chain_idx in range(num_chains):
            sampler = self._clone_sampler()
            chain, stats = self._run_single_chain(
                sampler,
                initial_xs[chain_idx],
                num_steps=num_steps,
                warmup=warmup,
                seed=seeds[chain_idx],
                progress=progress
            )
            chains.append(chain)
            stats_list.append(stats)

        chains_arr = np.stack(chains, axis=0)
        diagnostics: Dict[str, Any] = {}
        if compute_diagnostics and num_chains > 1:
            diagnostics["rhat_rank_normalized"] = compute_rhat_rank_normalized(chains_arr)
        return OrchestrationResult(chains=chains_arr, stats=stats_list, diagnostics=diagnostics)

    def _run_single_chain(
        self,
        sampler: DiffusionMH,
        initial_x: torch.Tensor,
        num_steps: int,
        warmup: int,
        seed: Optional[int],
        progress: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        state = sampler.initialize_state(initial_x, seed=seed)
        adapt_state = self._init_adapt_state(sampler)
        train_buffer: List[np.ndarray] = []
        train_updates = 0
        start_time = time.time()

        for step in range(warmup):
            state, info = sampler.step(state, stats)
            if self.adapt_config.enabled:
                adapt_state = self._update_adaptation(sampler, adapt_state, info, step + 1)
            if self.training_schedule.enabled and sampler.global_proposal is not None:
                train_buffer.append(state.x.detach().cpu().numpy())
                if len(train_buffer) > self.training_schedule.buffer_size:
                    train_buffer = train_buffer[-self.training_schedule.buffer_size:]
                if (step + 1) % self.training_schedule.train_every == 0 and train_updates < self.training_schedule.max_updates:
                    samples = np.array(train_buffer, dtype=np.float32)
                    train_flow_matching(
                        sampler.global_proposal,
                        samples,
                        batch_size=self.training_schedule.batch_size,
                        epochs=self.training_schedule.epochs,
                        lr=self.training_schedule.lr,
                        verbose=False
                    )
                    train_updates += 1

        chain = []
        for _ in range(num_steps):
            state, _ = sampler.step(state, stats)
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

    def _init_adapt_state(self, sampler: DiffusionMH) -> Dict[str, Any]:
        local_param = self._get_local_param(sampler.local_kernel)
        adapt_state = {
            "local_log_param": None,
            "p_global_logit": None,
            "window_attempts_local": 0,
            "window_accept_local": 0,
            "window_attempts_global": 0,
            "window_accept_global": 0
        }
        if local_param is not None:
            adapt_state["local_log_param"] = float(np.log(local_param))
        if sampler.p_global is not None:
            p = min(max(sampler.p_global, 1e-6), 1 - 1e-6)
            adapt_state["p_global_logit"] = float(np.log(p / (1 - p)))
        return adapt_state

    def _update_adaptation(
        self,
        sampler: DiffusionMH,
        adapt_state: Dict[str, Any],
        info: Dict[str, Any],
        step_idx: int
    ) -> Dict[str, Any]:
        if info["is_global"]:
            adapt_state["window_attempts_global"] += 1
            if info["accepted"]:
                adapt_state["window_accept_global"] += 1
        else:
            adapt_state["window_attempts_local"] += 1
            if info["accepted"]:
                adapt_state["window_accept_local"] += 1

        if step_idx % self.adapt_config.interval != 0:
            return adapt_state

        if adapt_state["window_attempts_local"] > 0:
            acc = adapt_state["window_accept_local"] / adapt_state["window_attempts_local"]
            if adapt_state["local_log_param"] is not None:
                adapt_state["local_log_param"] += self.adapt_config.adapt_rate * (acc - self.adapt_config.target_accept_local)
                new_param = float(np.exp(adapt_state["local_log_param"]))
                new_param = float(np.clip(new_param, self.adapt_config.min_local_step, self.adapt_config.max_local_step))
                self._set_local_param(sampler.local_kernel, new_param)

        if adapt_state["window_attempts_global"] > 0 and adapt_state["p_global_logit"] is not None:
            acc_g = adapt_state["window_accept_global"] / adapt_state["window_attempts_global"]
            adapt_state["p_global_logit"] += self.adapt_config.adapt_rate * (acc_g - self.adapt_config.target_accept_global)
            p = 1.0 / (1.0 + np.exp(-adapt_state["p_global_logit"]))
            p = float(np.clip(p, self.adapt_config.min_p_global, self.adapt_config.max_p_global))
            sampler.p_global = p

        adapt_state["window_attempts_local"] = 0
        adapt_state["window_accept_local"] = 0
        adapt_state["window_attempts_global"] = 0
        adapt_state["window_accept_global"] = 0
        return adapt_state

    def _get_local_param(self, kernel: Any) -> Optional[float]:
        if hasattr(kernel, "scale"):
            return float(kernel.scale)
        if hasattr(kernel, "step_size"):
            return float(kernel.step_size)
        return None

    def _set_local_param(self, kernel: Any, value: float) -> None:
        if hasattr(kernel, "scale"):
            kernel.scale = value
        if hasattr(kernel, "step_size"):
            kernel.step_size = value
