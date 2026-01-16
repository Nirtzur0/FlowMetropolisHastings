import numpy as np

from diffmcmc.diagnostics.sbc import run_sbc

def test_sbc_gaussian_posterior_rank_mean():
    def prior_sampler(rng):
        return rng.normal(0.0, 1.0)

    def simulate_data(theta, rng):
        return rng.normal(theta, 1.0)

    def posterior_sampler(y, rng, n):
        mu = 0.5 * y
        sigma = np.sqrt(0.5)
        return rng.normal(mu, sigma, size=n)

    num_replications = 50
    num_posterior_samples = 200
    ranks = run_sbc(
        prior_sampler,
        simulate_data,
        posterior_sampler,
        num_replications=num_replications,
        num_posterior_samples=num_posterior_samples,
        seed=123
    )
    mean_rank = np.mean(ranks)
    assert abs(mean_rank - num_posterior_samples / 2) < 25
