import numpy as np

from diffmcmc.core.acceptance import delayed_acceptance_log_alphas

def test_delayed_acceptance_detailed_balance_identity():
    rng = np.random.default_rng(0)
    log_pi_x, log_pi_xp, log_q_x, log_q_xp, log_qtilde_x, log_qtilde_xp = rng.normal(size=6)

    log_a1, log_a2 = delayed_acceptance_log_alphas(
        log_pi_x, log_pi_xp, log_q_x, log_q_xp, log_qtilde_x, log_qtilde_xp
    )
    log_a1_r, log_a2_r = delayed_acceptance_log_alphas(
        log_pi_xp, log_pi_x, log_q_xp, log_q_x, log_qtilde_xp, log_qtilde_x
    )

    log_lhs = log_pi_x + log_q_xp + log_a1 + log_a2
    log_rhs = log_pi_xp + log_q_x + log_a1_r + log_a2_r
    assert np.allclose(log_lhs, log_rhs, atol=1e-10)
