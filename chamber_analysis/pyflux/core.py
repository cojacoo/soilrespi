"""Main pyFlux analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .chamber import ChamberParams
from ..metadata import MeasurementMetadata, extract_metadata
from .regression import find_best_window
from .constants import GAS_META
from .flux_term import flux_term
from .models import lm_flux, hm_flux, LMResult, HMResult
from .selection import best_flux, FluxResult


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class SensorFluxResult:
    """Complete flux result for one sensor × one measurement file.

    Contains both the raw flux (in µmol m⁻² s⁻¹) and the value scaled to
    the sensor-appropriate reporting unit, plus all auxiliary metadata.
    """

    filename: str
    sensor: str
    flux_result: FluxResult

    # Regression window
    window_start_s: float
    window_end_s: float
    window_r2: float

    # Reported flux in sensor-appropriate unit
    flux_reported: float
    flux_se_reported: float
    report_unit: str

    # Environmental parameters used for the flux calculation
    temperature_C: float
    pressure_mbar: float

    # Auxiliary metadata
    metadata: MeasurementMetadata

    # ── Flat export ───────────────────────────────────────────────────────

    def to_flat_dict(self) -> dict:
        """Return a single-row dict suitable for building a summary DataFrame.

        Column groups
        -------------
        Best-model flux  : ``flux``, ``flux_se``, ``model``, ``quality``
        LM estimates     : ``flux_lm``, ``flux_se_lm``, ``slope_per_sec/min``, LM stats
        HM estimates     : ``flux_hm``, ``flux_se_hm``, ``hm_C0/Cinf/kappa``, HM stats
        Regression window: ``window_*``
        Environment      : T, P, GPS, soil, battery …
        """
        fr   = self.flux_result
        meta = self.metadata
        lm   = fr.lm
        hm   = fr.hm
        rf   = GAS_META.get(self.sensor, {}).get("report_factor", 1.0)
        hm_ok = hm is not None and hm.converged

        def _r(v, n=6):
            return round(v, n) if (v is not None and np.isfinite(v)) else None

        return {
            # ── Identity ──────────────────────────────────────────────────
            "filename":               self.filename,
            "sensor":                 self.sensor,
            "unit":                   self.report_unit,
            # ── Best-model flux (main result) ─────────────────────────────
            "model":                  fr.model,
            "flux":                   _r(self.flux_reported),
            "flux_se":                _r(self.flux_se_reported),
            "quality":                fr.quality,
            "g_factor":               _r(fr.g_factor, 4),
            # ── Best-model statistics ──────────────────────────────────────
            "r_squared":              _r(fr.r_squared, 4),
            "aicc":                   _r(fr.aicc, 3),
            "rmse":                   _r(fr.rmse, 6),
            "mae":                    _r(fr.mae, 6),
            "p_value":                _r(fr.p_value, 6),
            # ── LM individual estimates ───────────────────────────────────
            # flux_se_lm = SE(slope) × flux_term  [OLS standard error]
            "flux_lm":                _r(lm.flux * rf),
            "flux_se_lm":             _r(lm.flux_se * rf),
            "r_squared_lm":           _r(lm.r_squared, 4),
            "aicc_lm":                _r(lm.aicc, 3),
            "rmse_lm":                _r(lm.rmse, 6),
            "slope_per_sec":          _r(lm.slope, 9),
            "slope_per_min":          _r(lm.slope * 60.0, 7),
            # ── HM individual estimates ───────────────────────────────────
            "flux_hm":                _r(hm.flux * rf)      if hm_ok else None,
            "flux_se_hm":             _r(hm.flux_se * rf)   if hm_ok else None,
            "r_squared_hm":           _r(hm.r_squared, 4)   if hm_ok else None,
            "aicc_hm":                _r(hm.aicc, 3)        if hm_ok else None,
            "rmse_hm":                _r(hm.rmse, 6)        if hm_ok else None,
            "hm_C0":                  _r(hm.C0, 4)          if hm_ok else None,
            "hm_Cinf":                _r(hm.Cinf, 4)        if hm_ok else None,
            "hm_kappa":               _r(hm.kappa, 8)       if hm_ok else None,
            # ── Regression window ──────────────────────────────────────────
            "window_start_s":         _r(self.window_start_s, 1),
            "window_end_s":           _r(self.window_end_s, 1),
            "window_r2":              _r(self.window_r2, 4),
            # ── Environmental conditions ───────────────────────────────────
            "temperature_C":          _r(self.temperature_C, 2),
            "pressure_mbar":          _r(self.pressure_mbar, 2),
            # ── Timing ────────────────────────────────────────────────────
            "timestamp_start":        str(meta.timestamp_start),
            "timestamp_end":          str(meta.timestamp_end),
            "duration_s":             _r(meta.duration_s, 1),
            "n_samples":              meta.n_samples,
            # ── GPS ───────────────────────────────────────────────────────
            "gps_latitude":           meta.gps_latitude,
            "gps_longitude":          meta.gps_longitude,
            "gps_altitude_m":         meta.gps_altitude_m,
            # ── Atmospheric ───────────────────────────────────────────────
            "temperature_intern_C":   meta.temperature_intern_C,
            "humidity_intern_pct":    meta.humidity_intern_pct,
            "pressure_intern_mbar":   meta.pressure_intern_mbar,
            "temperature_extern_C":   meta.temperature_extern_C,
            "humidity_extern_pct":    meta.humidity_extern_pct,
            "pressure_extern_mbar":   meta.pressure_extern_mbar,
            # ── Soil ──────────────────────────────────────────────────────
            "soil_water_content_pct": meta.soil_water_content_pct,
            "soil_temperature_C":     meta.soil_temperature_C,
            "soil_permittivity":      meta.soil_permittivity,
            # ── SCD30 ─────────────────────────────────────────────────────
            "temperature_scd30_C":    meta.temperature_scd30_C,
            "humidity_scd30_pct":     meta.humidity_scd30_pct,
            # ── Device ────────────────────────────────────────────────────
            "battery_level_pct":      meta.battery_level_pct,
        }


# ── Internal helpers ──────────────────────────────────────────────────────


def _resolve_env(df: pd.DataFrame, params: ChamberParams) -> Tuple[float, float]:
    """Return (temperature_C, pressure_mbar) from data columns or fixed params."""
    T_col = "Temperature intern (degC)"
    P_col = "Pressure intern (mbar)"

    if params.temperature_C == "auto" and T_col in df.columns:
        T = float(df[T_col].dropna().mean())
    else:
        T = float(params.temperature_C)

    if params.pressure_mbar == "auto" and P_col in df.columns:
        P = float(df[P_col].replace(0.0, np.nan).dropna().mean())
    else:
        P = float(params.pressure_mbar)

    return T, P


# ── Public API ────────────────────────────────────────────────────────────


def analyze_sensor(
    df: pd.DataFrame,
    sensor: str,
    params: ChamberParams,
    filename: str = "",
    metadata: Optional[MeasurementMetadata] = None,
) -> Optional[SensorFluxResult]:
    """Run the full pyFlux pipeline for a single sensor column.

    Steps
    -----
    1. Extract environmental conditions (T, P) and chamber metadata.
    2. Convert native sensor units to ppm.
    3. Find the most linear window via sliding-window R² maximisation.
    4. Fit both LM and HM to the windowed data.
    5. Select the best model and assign a quality flag.
    6. Convert the flux to the sensor-appropriate reporting unit.

    Parameters
    ----------
    df       : DataFrame from :func:`~chamber_analysis.io.load_csv`
    sensor   : column name; must be a key in :data:`~chamber_analysis.pyflux.constants.GAS_META`
    params   : chamber physical parameters
    filename : source filename (used for labelling only)
    metadata : pre-extracted :class:`~chamber_analysis.metadata.MeasurementMetadata`;
               computed from ``df`` if not provided

    Returns
    -------
    :class:`SensorFluxResult`, or ``None`` if fewer than 10 valid data points.
    """
    if sensor not in GAS_META:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Register it in chamber_analysis.pyflux.constants.GAS_META."
        )
    if sensor not in df.columns:
        return None

    sensor_df = df[["time_sec", sensor]].dropna().copy()
    if len(sensor_df) < 10:
        return None

    meta   = metadata or extract_metadata(df, filename)
    T_C, P_mbar = _resolve_env(df, params)

    gas       = GAS_META[sensor]
    t         = sensor_df["time_sec"].values
    c_ppm     = sensor_df[sensor].values * gas["factor_to_ppm"]
    prec_ppm  = gas["precision"] * gas["factor_to_ppm"]

    # Sliding-window best linear segment
    i_start, i_end, win_r2 = find_best_window(t, c_ppm, params.window_frac, params.window_undercut)
    t_win = t[i_start:i_end] - t[i_start]   # zero-shift: HM requires t[0] == 0
    c_win = c_ppm[i_start:i_end]

    # Flux term (unit conversion factor)
    ft = flux_term(params.volume_L, params.area_m2, T_C, P_mbar)

    # Fit both models
    lm_res = lm_flux(t_win, c_win, ft, prec_ppm)
    hm_res = hm_flux(t_win, c_win, ft, prec_ppm, lm_slope=lm_res.slope)

    # Best-model selection + quality flag
    fr = best_flux(lm_res, hm_res, g_factor_max=params.g_factor_max)

    # Scale to reporting unit
    rf            = gas["report_factor"]
    flux_rep      = fr.flux * rf
    flux_se_rep   = fr.flux_se * rf if np.isfinite(fr.flux_se) else np.nan

    return SensorFluxResult(
        filename=filename,
        sensor=sensor,
        flux_result=fr,
        window_start_s=float(t[i_start]),
        window_end_s=float(t[i_end - 1]),
        window_r2=float(win_r2),
        flux_reported=float(flux_rep),
        flux_se_reported=float(flux_se_rep),
        report_unit=gas["report_unit"],
        temperature_C=T_C,
        pressure_mbar=P_mbar,
        metadata=meta,
    )


def analyze_file(
    df: pd.DataFrame,
    params: ChamberParams,
    sensors: Optional[List[str]] = None,
    filename: str = "",
) -> List[SensorFluxResult]:
    """Run pyFlux for all active sensors in a single measurement file.

    Parameters
    ----------
    sensors : sensor column names to process;
              defaults to all keys in :data:`~chamber_analysis.pyflux.constants.GAS_META`
              that are present in ``df``
    """
    active = sensors or [s for s in GAS_META if s in df.columns]
    meta   = extract_metadata(df, filename)
    return [
        r
        for s in active
        if (r := analyze_sensor(df, s, params, filename=filename, metadata=meta)) is not None
    ]


def analyze_directory(
    all_dfs: Dict[str, pd.DataFrame],
    params: ChamberParams,
    sensors: Optional[List[str]] = None,
) -> Tuple[List[SensorFluxResult], pd.DataFrame]:
    """Run pyFlux for every file and sensor in a loaded dataset.

    Parameters
    ----------
    all_dfs : filename → DataFrame mapping from
              :func:`~chamber_analysis.io.load_directory`

    Returns
    -------
    results    : flat list of :class:`SensorFluxResult`
    summary_df : tidy DataFrame, one row per file × sensor
    """
    all_results: List[SensorFluxResult] = []
    for fname, df in all_dfs.items():
        all_results.extend(analyze_file(df, params, sensors=sensors, filename=fname))

    summary_df = pd.DataFrame([r.to_flat_dict() for r in all_results])
    return all_results, summary_df
