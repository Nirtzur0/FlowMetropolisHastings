from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional
import warnings

class InferenceMode(str, Enum):
    EXACT = "exact"
    PSEUDO_MARGINAL = "pseudo_marginal"
    APPROX = "approx"

@dataclass
class InferenceContract:
    requested_mode: InferenceMode
    effective_mode: InferenceMode
    warnings: List[str]
    errors: List[str]

def _as_mode(mode: Any) -> InferenceMode:
    if isinstance(mode, InferenceMode):
        return mode
    if isinstance(mode, str):
        return InferenceMode(mode.lower())
    raise ValueError(f"Unsupported inference_mode: {mode!r}")

def validate_inference_mode(
    mode: Any,
    proposal: Optional[Any],
    strict: bool = False,
) -> InferenceContract:
    requested = _as_mode(mode)
    effective = requested
    warnings_list: List[str] = []
    errors_list: List[str] = []

    if proposal is None:
        if requested != InferenceMode.APPROX:
            errors_list.append("Global proposal is required for exact or pseudo-marginal modes.")
        return _finalize_contract(requested, effective, warnings_list, errors_list, strict)

    if requested == InferenceMode.EXACT:
        if not hasattr(proposal, "log_prob_exact"):
            errors_list.append("Exact mode requires proposal.log_prob_exact.")
        if not hasattr(proposal, "log_prob_cheap"):
            warnings_list.append("Exact mode without proposal.log_prob_cheap; stage-1 will use exact density.")
        if hasattr(proposal, "exactness_report"):
            warnings_list.extend([w for w in proposal.exactness_report() if w])
        elif hasattr(proposal, "is_exact") and not proposal.is_exact():
            warnings_list.append("Proposal reports non-exact density; exactness is not guaranteed.")

    if requested == InferenceMode.PSEUDO_MARGINAL:
        if not hasattr(proposal, "log_prob_unbiased"):
            warnings_list.append("Pseudo-marginal mode requires proposal.log_prob_unbiased; falling back to approx.")
            effective = InferenceMode.APPROX

    return _finalize_contract(requested, effective, warnings_list, errors_list, strict)

def _finalize_contract(
    requested: InferenceMode,
    effective: InferenceMode,
    warnings_list: List[str],
    errors_list: List[str],
    strict: bool,
) -> InferenceContract:
    if errors_list:
        if strict:
            raise ValueError("; ".join(errors_list))
        for msg in errors_list:
            warnings.warn(msg, RuntimeWarning)
        effective = InferenceMode.APPROX
    if warnings_list:
        if strict:
            raise ValueError("; ".join(warnings_list))
        for msg in warnings_list:
            warnings.warn(msg, RuntimeWarning)
    return InferenceContract(
        requested_mode=requested,
        effective_mode=effective,
        warnings=warnings_list,
        errors=errors_list,
    )
