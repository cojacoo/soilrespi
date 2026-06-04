"""Model selection and quality assessment — pyFlux equivalent of goFlux best.flux()."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .models import LMResult, HMResult


@dataclass
class FluxResult:
    """Best-model flux estimate for one sensor × one measurement file.

    Attributes
    ----------
    model     : winning model — ``"LM"`` or ``"HM"``
    flux      : best-model flux [µmol m⁻² s⁻¹]
                (multiply by ``report_factor`` for the reporting unit)
    flux_se   : standard error of the flux estimate
    r_squared : R² of the winning model
    aicc      : AICc of the winning model
    rmse      : RMSE of the winning model [ppm]
    mae       : MAE of the winning model [ppm]
    p_value   : LM slope p-value (most conservative significance test)
    quality   : one of ``"good"``, ``"below_mdf"``, ``"below_detection"``
    g_factor  : |HM flux / LM flux|; curvature indicator (NaN if HM failed)
    lm        : full LM result
    hm        : full HM result (may have ``converged=False``)
    """

    model: str
    flux: float
    flux_se: float
    r_squared: float
    aicc: float
    rmse: float
    mae: float
    p_value: float
    quality: str
    g_factor: float
    lm: LMResult
    hm: Optional[HMResult]


def best_flux(
    lm: LMResult,
    hm: Optional[HMResult],
    *,
    g_factor_max: float = 4.0,
    p_threshold: float = 0.05,
) -> FluxResult:
    """Select the best-fit model and assign a quality flag.

    Selection algorithm (mirrors goFlux ``best.flux()``):

    1. If HM did not converge or its g-factor exceeds ``g_factor_max``,
       use LM unconditionally.
    2. Otherwise score both models: the worse model on each of the four
       criteria (MAE, RMSE, AICc, SE) earns one penalty point.
    3. The model with fewer penalty points wins. Ties go to HM
       (non-linear model preferred when equally good).

    Quality flags (checked in order of priority):

    * ``"below_detection"`` – LM p-value > ``p_threshold``
    * ``"below_mdf"``       – |flux| < minimal detectable flux
    * ``"good"``            – all checks passed

    Parameters
    ----------
    lm           : result from :func:`~chamber_analysis.pyflux.models.lm_flux`
    hm           : result from :func:`~chamber_analysis.pyflux.models.hm_flux`
    g_factor_max : maximum acceptable |HM / LM| flux ratio before falling back to LM
    p_threshold  : significance threshold for the "below detection" flag
    """
    # ── g-factor ─────────────────────────────────────────────────────────
    hm_ok = hm is not None and hm.converged
    if hm_ok and abs(lm.flux) > 1e-12:
        g_factor = abs(hm.flux / lm.flux)
    else:
        g_factor = np.nan

    # ── Model selection ───────────────────────────────────────────────────
    use_hm = hm_ok and np.isfinite(g_factor) and g_factor <= g_factor_max

    if not use_hm:
        chosen_model = "LM"
        chosen = lm
    else:
        lm_score = hm_score = 0
        for lm_val, hm_val in [
            (lm.mae,  hm.mae),
            (lm.rmse, hm.rmse),
            (lm.aicc, hm.aicc),
            (lm.flux_se, hm.flux_se),
        ]:
            if np.isfinite(lm_val) and np.isfinite(hm_val):
                if lm_val > hm_val:
                    lm_score += 1
                elif hm_val > lm_val:
                    hm_score += 1
        # Ties go to HM
        chosen_model = "LM" if lm_score < hm_score else "HM"
        chosen = lm if chosen_model == "LM" else hm

    # ── Quality flag ──────────────────────────────────────────────────────
    if lm.p_value > p_threshold:
        quality = "below_detection"
    elif np.isfinite(chosen.flux) and np.isfinite(chosen.mdf) and abs(chosen.flux) < chosen.mdf:
        quality = "below_mdf"
    else:
        quality = "good"

    return FluxResult(
        model=chosen_model,
        flux=float(chosen.flux),
        flux_se=float(chosen.flux_se) if np.isfinite(chosen.flux_se) else np.nan,
        r_squared=float(chosen.r_squared),
        aicc=float(chosen.aicc),
        rmse=float(chosen.rmse),
        mae=float(chosen.mae),
        p_value=float(lm.p_value),
        quality=quality,
        g_factor=float(g_factor) if np.isfinite(g_factor) else np.nan,
        lm=lm,
        hm=hm,
    )
