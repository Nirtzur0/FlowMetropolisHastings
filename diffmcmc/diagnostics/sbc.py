from typing import Callable, List, Optional
import numpy as np

def run_sbc(
    prior_sampler: Callable[[np.random.Generator], float],
    simulate_data: Callable[[float, np.random.Generator], object],
    posterior_sampler: Callable[[object, np.random.Generator, int], np.ndarray],
    num_replications: int,
    num_posterior_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Run Simulation-Based Calibration (SBC).
    Returns rank statistics (size num_replications).
    """
    rng = np.random.default_rng(seed)
    ranks: List[int] = []
    for _ in range(num_replications):
        theta = prior_sampler(rng)
        data = simulate_data(theta, rng)
        draws = posterior_sampler(data, rng, num_posterior_samples)
        rank = int(np.sum(draws < theta))
        ranks.append(rank)
    return np.array(ranks)

def sbc_uniformity_check(ranks: np.ndarray, num_posterior_samples: int) -> float:
    """
    Compute a simple chi-square statistic for SBC ranks.
    """
    bins = np.arange(0, num_posterior_samples + 2) - 0.5
    hist, _ = np.histogram(ranks, bins=bins)
    expected = np.full_like(hist, ranks.size / hist.size, dtype=np.float64)
    chi2 = np.sum((hist - expected) ** 2 / (expected + 1e-9))
    return float(chi2)
