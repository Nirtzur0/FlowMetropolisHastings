from typing import Tuple

def delayed_acceptance_log_r1(
    log_pi_x: float,
    log_pi_xp: float,
    log_qtilde_x: float,
    log_qtilde_xp: float,
) -> float:
    """
    Stage-1 delayed acceptance log-ratio using a cheap surrogate proposal density.
    """
    return log_pi_xp - log_pi_x + log_qtilde_x - log_qtilde_xp

def delayed_acceptance_log_r2(
    log_q_x: float,
    log_q_xp: float,
    log_qtilde_x: float,
    log_qtilde_xp: float,
) -> float:
    """
    Stage-2 correction log-ratio for delayed acceptance.
    """
    return log_q_x - log_q_xp + log_qtilde_xp - log_qtilde_x

def delayed_acceptance_log_alphas(
    log_pi_x: float,
    log_pi_xp: float,
    log_q_x: float,
    log_q_xp: float,
    log_qtilde_x: float,
    log_qtilde_xp: float,
) -> Tuple[float, float]:
    """
    Compute log(alpha1), log(alpha2) for delayed-acceptance independence MH.
    """
    log_r1 = delayed_acceptance_log_r1(log_pi_x, log_pi_xp, log_qtilde_x, log_qtilde_xp)
    log_r2 = delayed_acceptance_log_r2(log_q_x, log_q_xp, log_qtilde_x, log_qtilde_xp)
    return min(0.0, log_r1), min(0.0, log_r2)
