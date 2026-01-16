import argparse
import json
import os
import tomllib
from typing import Any, Dict, Tuple

import numpy as np
import torch

from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel, MALAKernel
from diffmcmc.core.orchestrator import SamplerOrchestrator, AdaptConfig, TrainingSchedule
from diffmcmc.experiment import ExperimentLogger
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.targets.mog import GaussianMixtureTarget
from diffmcmc.targets.banana import BananaTarget

def _load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".toml"):
        with open(path, "rb") as f:
            return tomllib.load(f)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Config must be .toml or .json")

def _build_target(cfg: Dict[str, Any]) -> Tuple[Any, int]:
    t_cfg = cfg.get("target", {})
    t_type = t_cfg.get("type", "mog")
    if t_type == "mog":
        dim = int(t_cfg.get("dim", 2))
        centers = t_cfg.get("centers", None)
        scale = float(t_cfg.get("scale", 1.0))
        target = GaussianMixtureTarget(dim=dim, centers=centers, scale=scale)
        return target.log_prob, dim
    if t_type == "banana":
        dim = int(t_cfg.get("dim", 2))
        b = float(t_cfg.get("b", 0.1))
        target = BananaTarget(dim=dim, b=b)
        return target.log_prob, dim
    if t_type == "gaussian":
        dim = int(t_cfg.get("dim", 1))
        def log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1) - 0.5 * dim * np.log(2 * np.pi)
        return log_prob, dim
    raise ValueError(f"Unsupported target type: {t_type}")

def _build_local_kernel(cfg: Dict[str, Any]) -> Any:
    k_cfg = cfg.get("local_kernel", {})
    k_type = k_cfg.get("type", "rwm")
    if k_type == "rwm":
        return RWMKernel(scale=float(k_cfg.get("scale", 1.0)))
    if k_type == "mala":
        return MALAKernel(step_size=float(k_cfg.get("step_size", 0.1)))
    raise ValueError(f"Unsupported local kernel type: {k_type}")

def _build_global_proposal(cfg: Dict[str, Any], dim: int) -> Any:
    g_cfg = cfg.get("global_proposal", {})
    if not g_cfg or g_cfg.get("type", "flow") is None:
        return None
    if g_cfg.get("type", "flow") != "flow":
        raise ValueError("Only flow global_proposal is supported in CLI.")
    return FlowProposal(
        dim=dim,
        step_size=float(g_cfg.get("step_size", 0.1)),
        deterministic_trace=bool(g_cfg.get("deterministic_trace", True)),
        mixture_prob=float(g_cfg.get("mixture_prob", 0.0)),
        broad_scale=float(g_cfg.get("broad_scale", 2.0)),
        integrator=str(g_cfg.get("integrator", "rk4")),
        enforce_divisible=bool(g_cfg.get("enforce_divisible", True)),
        auto_adjust_step_size=bool(g_cfg.get("auto_adjust_step_size", True)),
        exact_discrete_logdet=bool(g_cfg.get("exact_discrete_logdet", False))
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="DiffMCMC config-driven runner")
    parser.add_argument("--config", required=True, help="Path to .toml or .json config file")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    log_prob_fn, dim = _build_target(cfg)
    local_kernel = _build_local_kernel(cfg)
    global_proposal = _build_global_proposal(cfg, dim)

    sampler_cfg = cfg.get("sampler", {})
    sampler = DiffusionMH(
        log_prob_fn=log_prob_fn,
        dim=dim,
        local_kernel=local_kernel,
        global_proposal=global_proposal,
        p_global=float(sampler_cfg.get("p_global", 0.2)),
        device=str(sampler_cfg.get("device", "cpu")),
        inference_mode=str(sampler_cfg.get("inference_mode", "approx")),
        strict_exactness=bool(sampler_cfg.get("strict_exactness", False))
    )

    exp_cfg = cfg.get("experiment", {})
    output_dir = args.output_dir or exp_cfg.get("output_dir", "runs")
    logger = ExperimentLogger(root_dir=output_dir, name=exp_cfg.get("name", None))
    logger.save_config(cfg)

    num_steps = int(sampler_cfg.get("num_steps", 1000))
    warmup = int(sampler_cfg.get("warmup", 200))
    num_chains = int(cfg.get("orchestrator", {}).get("num_chains", 1))

    if num_chains > 1:
        adapt_cfg = AdaptConfig(**cfg.get("adapt", {}))
        train_cfg = TrainingSchedule(**cfg.get("training", {}))
        orchestrator = SamplerOrchestrator(
            sampler=sampler,
            adapt_config=adapt_cfg,
            training_schedule=train_cfg,
            share_global_proposal=bool(cfg.get("orchestrator", {}).get("share_global_proposal", False))
        )
        init_xs = [torch.zeros(dim) for _ in range(num_chains)]
        seeds = exp_cfg.get("seeds", None)
        if seeds is None and exp_cfg.get("seed", None) is not None:
            base_seed = int(exp_cfg.get("seed"))
            seeds = [base_seed + i for i in range(num_chains)]
        result = orchestrator.run_chains(
            init_xs,
            num_steps=num_steps,
            warmup=warmup,
            seeds=seeds,
            progress=True
        )
        logger.save_chain(result.chains, "chains")
        logger.save_stats(result.stats)
        if result.diagnostics:
            logger.save_diagnostics(result.diagnostics)
    else:
        initial_x = torch.zeros(dim)
        chain, stats = sampler.run(initial_x, num_steps=num_steps, warmup=warmup, seed=exp_cfg.get("seed", None))
        logger.save_chain(chain, "chain")
        logger.save_stats(stats)

if __name__ == "__main__":
    main()
