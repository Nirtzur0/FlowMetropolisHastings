import numpy as np

from diffmcmc.diagnostics.metrics import compute_ess, compute_rhat_rank_normalized

def test_compute_ess_constant_chain():
    chain = np.zeros((100, 2))
    ess = compute_ess(chain)
    assert np.all(ess == 1.0)

def test_rhat_rank_normalized_identical_chains():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 200, 3))
    chains = np.repeat(base, 4, axis=0)
    rhat = compute_rhat_rank_normalized(chains)
    assert np.allclose(rhat, 1.0, atol=1e-6)
