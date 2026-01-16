import numpy as np
import scipy.stats

def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.array([])
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom == 0:
        return np.zeros(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conjugate(f)))[:n]
    acf = acf / acf[0]
    return acf

def compute_ess(chain: np.ndarray) -> np.ndarray:
    """
    Compute Effective Sample Size (ESS) per dimension using Geyer's initial
    monotone sequence estimator.
    """
    N, D = chain.shape
    if N < 3:
        return np.ones(D)
    ess = np.zeros(D)
    for d in range(D):
        x = chain[:, d]
        acf = _autocorr_fft(x)
        if acf.size == 0 or acf[0] == 0:
            ess[d] = 1.0
            continue
        pair_sums = []
        for k in range(1, len(acf) - 1, 2):
            pair = acf[k] + acf[k + 1]
            if pair < 0:
                break
            pair_sums.append(pair)
        for i in range(1, len(pair_sums)):
            if pair_sums[i] > pair_sums[i - 1]:
                pair_sums[i] = pair_sums[i - 1]
        tau = 1.0 + 2.0 * np.sum(pair_sums)
        ess[d] = max(1.0, N / tau)
    return ess

def compute_rhat_rank_normalized(chains: np.ndarray) -> np.ndarray:
    """
    Rank-normalized split R-hat (Vehtari et al. 2021).
    chains: (M, N, D) array.
    Returns: (D,) array.
    """
    if chains.ndim != 3:
        raise ValueError("chains must have shape (M, N, D)")
    m, n, d = chains.shape
    if n < 4 or m < 2:
        return np.ones(d)

    n2 = n // 2
    split = np.concatenate([chains[:, :n2, :], chains[:, n2:2*n2, :]], axis=0)
    m2, n2, d = split.shape

    flat = split.reshape(m2 * n2, d)
    ranks = np.vstack([
        scipy.stats.rankdata(flat[:, j], method="average") for j in range(d)
    ]).T
    z = scipy.stats.norm.ppf((ranks - 0.5) / (m2 * n2))
    z = z.reshape(m2, n2, d)

    chain_means = z.mean(axis=1)
    chain_vars = z.var(axis=1, ddof=1)
    B = n2 * chain_means.var(axis=0, ddof=1)
    W = chain_vars.mean(axis=0)
    var_hat = (n2 - 1) / n2 * W + B / n2
    rhat = np.sqrt(var_hat / W)
    rhat = np.where(np.isfinite(rhat), rhat, np.ones_like(rhat))
    return rhat
