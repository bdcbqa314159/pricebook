"""Black-76 — the option formula on a forward (L0 numerical toolkit).

The shared pricing primitive behind Garman-Kohlhagen (FX) and Black-Scholes
(equity): a European option on a forward `F`, struck at `K`, with total vol
`std = sigma*sqrt(T)`, discounted by `discount`:

    d1 = [ln(F/K) + std^2/2] / std,   d2 = d1 - std
    call = discount * (F N(d1) - K N(d2)),  put = discount * (K N(-d2) - F N(-d1))

`std <= 0` (expiry / zero vol) returns the discounted intrinsic. Returns the
option value per unit notional.

Provenance:
  quarry: python/pricebook/pricing/ (black)
  source: Black (1976)
  oracle: exercised by the FX (GK) and equity (BS) option oracles
  slice:  equity-option
"""

from __future__ import annotations

import math

from pricebook_ng.foundation.distributions import norm_cdf


def black_76(forward: float, strike: float, discount: float, std: float, is_call: bool) -> float:
    """European option value per unit notional on `forward` (Black 1976)."""
    if std < 1e-15:
        intrinsic = (forward - strike) if is_call else (strike - forward)
        return discount * max(intrinsic, 0.0)
    d1 = (math.log(forward / strike) + 0.5 * std * std) / std
    d2 = d1 - std
    if is_call:
        return discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    return discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
