"""Linear (LM) and Hutchinson-Mosier (HM) flux models."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit

from .flux_term import mdf as _mdf


# ── Result dataclasses ────────────────────────────────────────────────────


@dataclass
class LMResult:
    """Output of the linear flux model (LM)."""

    slope: float        # concentration slope [ppm s⁻¹]
    intercept: float    # [ppm]
    r_squared: float
    se: float           # standard error of slope [ppm s⁻¹]
    p_value: float      # two-tailed p-value for H₀: slope = 0
    aicc: float         # corrected Akaike information criterion
    rmse: float         # root mean squared error [ppm]
    mae: float          # mean absolute error [ppm]
    flux: float         # [µmol m⁻² s⁻¹]
    flux_se: float      # standard error of flux [µmol m⁻² s⁻¹]
    mdf: float          # minimal detectable flux [µmol m⁻² s⁻¹]
    n: int              # number of observations in the window


@dataclass
class HMResult:
    """Output of the Hutchinson-Mosier non-linear model (HM).

    Model: C(t) = C₀ + (C∞ − C₀)(1 − exp(−κ t))
    Flux : F = (C∞ − C₀) × κ × flux_term
    """

    C0: float           # initial concentration [ppm]
    Cinf: float         # asymptotic concentration [ppm]
    kappa: float        # rate constant [s⁻¹]
    r_squared: float
    se: float           # standard error of flux [µmol m⁻² s⁻¹]
    aicc: float
    rmse: float
    mae: float
    flux: float         # [µmol m⁻² s⁻¹]
    flux_se: float
    mdf: float
    n: int
    converged: bool     # False if scipy.optimize.curve_fit failed


# ── Internal helpers ──────────────────────────────────────────────────────


def _aicc(rss: float, n: int, k: int) -> float:
    """AICc for ordinary least-squares models.

    AIC  = n ln(RSS/n) + 2k
    AICc = AIC + 2k(k+1) / (n − k − 1)
    """
    if rss <= 0 or n <= k + 1:
        return np.inf
    aic = n * np.log(rss / n) + 2.0 * k
    return aic + 2.0 * k * (k + 1) / (n - k - 1)


def _hm_curve(t: np.ndarray, C0: float, Cinf: float, kappa: float) -> np.ndarray:
    return C0 + (Cinf - C0) * (1.0 - np.exp(-kappa * t))


# ── Public model functions ────────────────────────────────────────────────


def lm_flux(
    t: np.ndarray,
    c: np.ndarray,
    ft: float,
    precision_ppm: float,
) -> LMResult:
    """Fit a linear model to the windowed concentration series.

    Parameters
    ----------
    t             : elapsed time [s], shape (n,)
    c             : concentration [ppm], shape (n,)
    ft            : flux term from :func:`~chamber_analysis.pyflux.flux_term.flux_term`
    precision_ppm : instrument precision in ppm (for MDF calculation)
    """
    n = len(t)
    reg = stats.linregress(t, c)
    slope     = float(reg.slope)
    intercept = float(reg.intercept)
    se        = float(reg.stderr)
    p_value   = float(reg.pvalue)
    r_squared = float(reg.rvalue ** 2)

    fitted    = intercept + slope * t
    residuals = c - fitted
    rss       = float(np.sum(residuals ** 2))
    rmse      = float(np.sqrt(rss / n))
    mae       = float(np.mean(np.abs(residuals)))
    aicc      = _aicc(rss, n, k=2)

    duration = float(t[-1] - t[0])
    return LMResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        se=se,
        p_value=p_value,
        aicc=aicc,
        rmse=rmse,
        mae=mae,
        flux=slope * ft,
        flux_se=se * ft,
        mdf=_mdf(precision_ppm, duration, ft),
        n=n,
    )


def hm_flux(
    t: np.ndarray,
    c: np.ndarray,
    ft: float,
    precision_ppm: float,
    lm_slope: float,
) -> HMResult:
    """Fit the Hutchinson-Mosier non-linear model.

    Initial parameter guesses are derived from the linear-model slope
    and the observed concentration range. Bounds on κ prevent the model
    from fitting physically unreasonable curvatures (half-life < ~14 % of
    the measurement duration).

    Parameters
    ----------
    t             : elapsed time [s], shape (n,); should start near 0
    c             : concentration [ppm], shape (n,)
    ft            : flux term
    precision_ppm : instrument precision in ppm
    lm_slope      : slope from :func:`lm_flux` used for the κ initial guess
    """
    n        = len(t)
    duration = float(t[-1] - t[0])
    c_range  = float(np.ptp(c))

    # ── Initial guesses ───────────────────────────────────────────────────
    C0_init    = float(c[0])
    delta_init = lm_slope * duration          # expected total rise
    Cinf_init  = C0_init + delta_init * 1.5  # slightly beyond observed change
    denom      = max(abs(Cinf_init - C0_init), 1e-6)
    kappa_init = abs(lm_slope) / denom

    # ── Parameter bounds ─────────────────────────────────────────────────
    # C0   : allow ±slack around the observed range
    # Cinf : unconstrained in the direction of flux
    # kappa: positive; upper bound keeps half-life ≥ ~14 % of duration
    slack   = max(c_range, 10.0)
    kap_max = 10.0 / max(duration, 1.0)
    lo = [float(c.min()) - slack, -np.inf, 0.0    ]
    hi = [float(c.max()) + slack,  np.inf, kap_max]

    _FAIL = HMResult(
        C0=np.nan, Cinf=np.nan, kappa=np.nan,
        r_squared=np.nan, se=np.nan, aicc=np.inf,
        rmse=np.nan, mae=np.nan,
        flux=np.nan, flux_se=np.nan, mdf=np.nan,
        n=n, converged=False,
    )

    try:
        popt, pcov = curve_fit(
            _hm_curve, t, c,
            p0=[C0_init, Cinf_init, kappa_init],
            bounds=(lo, hi),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return _FAIL

    C0, Cinf, kappa = popt

    fitted    = _hm_curve(t, C0, Cinf, kappa)
    residuals = c - fitted
    rss       = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((c - c.mean()) ** 2))
    r_squared = float(1.0 - rss / ss_tot) if ss_tot > 0 else 0.0
    rmse      = float(np.sqrt(rss / n))
    mae       = float(np.mean(np.abs(residuals)))
    aicc      = _aicc(rss, n, k=3)

    # SE of flux via the delta method.
    # flux = (Cinf − C0) × κ × ft
    # Jacobian w.r.t. [C0, Cinf, κ]:
    #   ∂F/∂C0   = −κ × ft
    #   ∂F/∂Cinf =  κ × ft
    #   ∂F/∂κ    = (Cinf − C0) × ft
    J = np.array([-kappa * ft, kappa * ft, (Cinf - C0) * ft])
    try:
        flux_var = float(J @ pcov @ J.T)
        flux_se  = float(np.sqrt(max(0.0, flux_var)))
    except Exception:
        flux_se = np.nan

    return HMResult(
        C0=float(C0),
        Cinf=float(Cinf),
        kappa=float(kappa),
        r_squared=r_squared,
        se=flux_se,
        aicc=aicc,
        rmse=rmse,
        mae=mae,
        flux=float((Cinf - C0) * kappa * ft),
        flux_se=flux_se,
        mdf=_mdf(precision_ppm, duration, ft),
        n=n,
        converged=True,
    )
