"""Interactive Plotly visualizations for chamber analysis.

All functions return a ``plotly.graph_objects.Figure`` — no side effects.
Use directly in Streamlit with ``st.plotly_chart(fig, use_container_width=True)``
or in notebooks with ``fig.show()``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pyflux.constants import GAS_META
from .pyflux.core import SensorFluxResult

# 24-colour qualitative palette (replaces matplotlib tab20)
_PALETTE = px.colors.qualitative.Dark24

_LM_COLOR   = "#2ca02c"   # green
_HM_COLOR   = "#ff7f0e"   # orange
_DATA_COLOR = "#1f77b4"   # blue
_FIT_COLOR  = "#d62728"   # red
_WIN_FILL   = "rgba(255, 215, 0, 0.20)"


# ── Helpers ───────────────────────────────────────────────────────────────


def _quality_symbol(quality: str) -> str:
    return "✓" if quality == "good" else ("~" if quality == "below_mdf" else "!")


def _quality_color(quality: str) -> str:
    return "green" if quality == "good" else ("goldenrod" if quality == "below_mdf" else "red")


# ── Public functions ──────────────────────────────────────────────────────


def time_series_fig(
    df: pd.DataFrame,
    sensors: List[str],
    title: str = "",
    battery_col: str = "BatteryLevel (%)",
) -> go.Figure:
    """Multi-panel time-series for a set of sensor columns.

    Battery level is appended as an extra panel when available.
    """
    cols = [s for s in sensors if s in df.columns]
    if battery_col in df.columns and battery_col not in cols:
        cols.append(battery_col)
    if not cols:
        fig = go.Figure()
        fig.add_annotation(text="No data columns found", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    n   = len(cols)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=cols,
    )

    for i, col in enumerate(cols, 1):
        sub = df[["Timestamp", col]].dropna()
        fig.add_trace(
            go.Scatter(
                x=sub["Timestamp"], y=sub[col],
                mode="lines", name=col,
                line=dict(width=1.5, color=_DATA_COLOR if i < n else "#9467bd"),
                showlegend=False,
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text=col, title_font_size=10, row=i, col=1)

    fig.update_xaxes(title_text="Time", row=n, col=1)
    fig.update_layout(
        template="none",
        title_text=title or None,
        height=280 * n,
        margin=dict(t=60 if title else 30, b=40),
    )
    return fig


def regression_fig(
    result: SensorFluxResult,
    df: pd.DataFrame,
) -> go.Figure:
    """Single-sensor regression: full series, analysis window, LM and HM fits."""
    gas    = GAS_META.get(result.sensor, {})
    factor = gas.get("factor_to_ppm", 1.0)

    sdf   = df[["Timestamp", "time_sec", result.sensor]].dropna().copy()
    t     = sdf["time_sec"].values
    c_ppm = sdf[result.sensor].values * factor
    ts    = sdf["Timestamp"].values

    fr = result.flux_result
    lm = fr.lm

    win_mask  = (t >= result.window_start_s) & (t <= result.window_end_s)
    ts_win    = ts[win_mask]
    t_win     = t[win_mask]
    fitted_lm = lm.intercept + lm.slope * (t_win - t_win[0])

    fig = go.Figure()

    # Full time series (faded)
    fig.add_trace(go.Scatter(
        x=ts, y=c_ppm,
        mode="lines", name="Measurements",
        line=dict(color=_DATA_COLOR, width=1.5),
        opacity=0.55,
    ))

    # Analysis window background
    fig.add_vrect(
        x0=str(ts_win[0]), x1=str(ts_win[-1]),
        fillcolor="gold", opacity=0.18,
        layer="below", line_width=0,
        annotation_text="Analysis window",
        annotation_position="top left",
        annotation_font_size=10,
    )

    # LM fit line
    fig.add_trace(go.Scatter(
        x=ts_win, y=fitted_lm,
        mode="lines", name="LM fit",
        line=dict(color=_FIT_COLOR, width=2, dash="dash"),
    ))

    # HM fit curve (if converged)
    if fr.hm is not None and fr.hm.converged:
        hm   = fr.hm
        t0   = t_win[0]
        c_hm = hm.C0 + (hm.Cinf - hm.C0) * (1.0 - np.exp(-hm.kappa * (t_win - t0)))
        fig.add_trace(go.Scatter(
            x=ts_win, y=c_hm,
            mode="lines", name="HM fit",
            line=dict(color=_HM_COLOR, width=2, dash="dot"),
        ))

    unit = gas.get("report_unit", "µmol m⁻² s⁻¹")
    fig.update_layout(
        template="none",
        title=dict(
            text=(f"<b>{result.sensor}</b> – {result.filename}<br>"
                  f"<sup>Model: {fr.model} | "
                  f"Flux: {result.flux_reported:.5f} {unit} | "
                  f"R²={fr.r_squared:.4f} | "
                  f"p={fr.p_value:.4f} | "
                  f"quality: {fr.quality}</sup>"),
            font_size=13,
        ),
        xaxis_title="Time",
        yaxis_title=result.sensor,
        legend=dict(orientation="h", y=-0.15),
        height=420,
        margin=dict(t=80, b=80),
    )
    return fig


def flux_bar_fig(
    results: List[SensorFluxResult],
    sensor: str,
) -> go.Figure:
    """Bar chart of flux estimates across all measurement files for one sensor."""
    sub = [r for r in results if r.sensor == sensor]
    if not sub:
        fig = go.Figure()
        fig.add_annotation(text=f"No results for {sensor}", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    labels   = [r.filename.replace(".csv", "") for r in sub]
    fluxes   = [r.flux_reported for r in sub]
    ses      = [r.flux_se_reported if np.isfinite(r.flux_se_reported) else 0.0 for r in sub]
    colors   = [_LM_COLOR if r.flux_result.model == "LM" else _HM_COLOR for r in sub]
    symbols  = [_quality_symbol(r.flux_result.quality) for r in sub]
    sym_cols = [_quality_color(r.flux_result.quality) for r in sub]

    hover = [
        (f"<b>{r.filename}</b><br>"
         f"Flux: {r.flux_reported:.5f} {r.report_unit}<br>"
         f"SE: ±{r.flux_se_reported:.5f}<br>"
         f"Model: {r.flux_result.model}<br>"
         f"R²: {r.flux_result.r_squared:.4f}<br>"
         f"Quality: {r.flux_result.quality}")
        for r in sub
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=fluxes,
        error_y=dict(type="data", array=ses, visible=True, color="black", thickness=1.5),
        marker_color=colors,
        opacity=0.80,
        hovertext=hover,
        hoverinfo="text",
        name="Flux",
    ))

    # Quality markers as text above/below bars
    for i, (sym, col) in enumerate(zip(symbols, sym_cols)):
        fig.add_annotation(
            x=labels[i],
            y=fluxes[i] + (ses[i] if fluxes[i] >= 0 else -ses[i]),
            text=sym,
            showarrow=False,
            font=dict(size=12, color=col),
            yshift=8 if fluxes[i] >= 0 else -8,
        )

    fig.add_hline(y=0, line_color="black", line_width=0.8)
    unit = sub[0].report_unit
    fig.update_layout(
        template="none",
        title=f"<b>pyFlux – {sensor}</b>",
        xaxis_title="Measurement",
        yaxis_title=f"Flux ({unit})",
        xaxis_tickangle=-45,
        height=500,
        legend=dict(orientation="h", y=-0.25),
        annotations=[
            *fig.layout.annotations,
            dict(x=1.0, y=-0.30, xref="paper", yref="paper", showarrow=False,
                 text="<span style='color:green'>■</span> LM &nbsp;"
                      "<span style='color:orange'>■</span> HM &nbsp;"
                      "✓ good &nbsp; ~ below MDF &nbsp; ! below detection",
                 font_size=10, align="right"),
        ],
    )
    return fig


def overview_fig(
    results: List[SensorFluxResult],
    all_dfs: Dict[str, pd.DataFrame],
) -> go.Figure:
    """All measurements overlaid per sensor with regression lines — one subplot per sensor."""
    sensors    = sorted({r.sensor for r in results})
    all_fnames = sorted({r.filename for r in results})
    n_s        = len(sensors)

    if n_s == 0:
        fig = go.Figure()
        fig.add_annotation(text="No results", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=n_s, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=sensors,
    )

    for s_idx, sensor in enumerate(sensors, 1):
        gas    = GAS_META.get(sensor, {})
        factor = gas.get("factor_to_ppm", 1.0)
        unit   = gas.get("report_unit", "µmol m⁻² s⁻¹")

        for f_idx, fname in enumerate(all_fnames):
            df  = all_dfs.get(fname)
            r   = next((x for x in results if x.filename == fname and x.sensor == sensor), None)
            if df is None or r is None or sensor not in df.columns:
                continue

            color  = _PALETTE[f_idx % len(_PALETTE)]
            t_min  = df["time_sec"].values / 60.0
            c_ppm  = df[sensor].dropna().values * factor
            n      = min(len(t_min), len(c_ppm))
            label  = fname.replace(".csv", "")

            # Full series (faded)
            fig.add_trace(go.Scatter(
                x=t_min[:n], y=c_ppm[:n],
                mode="lines", name=label,
                line=dict(color=color, width=1.0),
                opacity=0.35,
                legendgroup=label,
                showlegend=(s_idx == 1),
                hovertemplate=f"{label}<br>t=%{{x:.1f}} min<br>c=%{{y:.2f}}<extra></extra>",
            ), row=s_idx, col=1)

            # Regression line in window
            lm     = r.flux_result.lm
            t_ends = np.array([r.window_start_s, r.window_end_s]) / 60.0
            c_fit  = lm.intercept + lm.slope * np.array([0.0, r.window_end_s - r.window_start_s])
            fig.add_trace(go.Scatter(
                x=t_ends, y=c_fit,
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash="dash"),
                legendgroup=label,
                showlegend=False,
                hovertemplate=(f"{label}<br>"
                               f"Flux={r.flux_reported:.4f} {unit}<br>"
                               f"R²={r.window_r2:.3f}<extra></extra>"),
            ), row=s_idx, col=1)

        fig.update_yaxes(title_text=sensor, title_font_size=10, row=s_idx, col=1)
        fig.update_xaxes(title_text="Time (min)", row=s_idx, col=1)

    fig.update_layout(
        template="none",
        title="<b>Overview – all measurements</b>",
        height=450 * n_s,
        legend=dict(
            orientation="v", x=1.02, y=1,
            title_text="Measurement",
            font_size=9,
        ),
        margin=dict(r=200),
    )
    return fig


def battery_fig(all_dfs: Dict[str, pd.DataFrame]) -> go.Figure:
    """Battery level over measurement time for all files."""
    fig = go.Figure()
    for f_idx, (fname, df) in enumerate(all_dfs.items()):
        if "BatteryLevel (%)" not in df.columns:
            continue
        sub = df[["time_min", "BatteryLevel (%)"]].dropna()
        fig.add_trace(go.Scatter(
            x=sub["time_min"], y=sub["BatteryLevel (%)"],
            mode="lines", name=fname.replace(".csv", ""),
            line=dict(color=_PALETTE[f_idx % len(_PALETTE)], width=1.5),
        ))

    fig.update_layout(
        template="none",
        title="<b>Battery level – all measurements</b>",
        xaxis_title="Time (min)",
        yaxis_title="Battery level (%)",
        legend=dict(orientation="v", x=1.02, font_size=9),
        height=420,
        margin=dict(r=180),
    )
    return fig


def model_comparison_fig(
    results: List[SensorFluxResult],
    sensor: str,
) -> go.Figure:
    """LM vs HM flux scatter (1:1 line) + grouped bar — when HM converged."""
    sub = [r for r in results if r.sensor == sensor and r.flux_result.hm is not None
           and r.flux_result.hm.converged]
    if len(sub) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need ≥2 measurements with converged HM model",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return fig

    labels   = [r.filename.replace(".csv", "") for r in sub]
    unit     = sub[0].report_unit
    rf       = GAS_META[sensor]["report_factor"]
    flux_lm  = [r.flux_result.lm.flux * rf for r in sub]
    flux_hm  = [r.flux_result.hm.flux * rf for r in sub]
    se_hm    = [r.flux_result.hm.flux_se * rf
                if np.isfinite(r.flux_result.hm.flux_se) else 0.0
                for r in sub]
    models   = [r.flux_result.model for r in sub]
    dot_cols = [_HM_COLOR if m == "HM" else _LM_COLOR for m in models]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LM vs HM flux per measurement", "Correlation (1:1 line)"],
        column_widths=[0.55, 0.45],
    )

    # ── Grouped bar ───────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=labels, y=flux_lm, name="LM", marker_color=_LM_COLOR, opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels, y=flux_hm, name="HM",
        marker_color=_HM_COLOR, opacity=0.8,
        error_y=dict(type="data", array=se_hm, visible=True,
                     color="black", thickness=1.2),
    ), row=1, col=1)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text=f"Flux ({unit})", row=1, col=1)

    # ── 1:1 scatter ───────────────────────────────────────────────────────
    all_vals = flux_lm + flux_hm
    lim      = [min(all_vals) * 1.1, max(all_vals) * 1.1]
    fig.add_trace(go.Scatter(
        x=lim, y=lim,
        mode="lines", name="1:1 line",
        line=dict(color="black", dash="dash", width=1),
        showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=flux_lm, y=flux_hm,
        mode="markers+text",
        marker=dict(color=dot_cols, size=9, line=dict(color="black", width=0.5)),
        text=labels, textposition="top center", textfont_size=8,
        error_y=dict(type="data", array=se_hm, visible=True, color="gray", thickness=1),
        name="Measurements",
        hovertemplate="LM=%{x:.4f}<br>HM=%{y:.4f}<extra></extra>",
    ), row=1, col=2)

    valid = [(lm, hm) for lm, hm in zip(flux_lm, flux_hm)
             if np.isfinite(lm) and np.isfinite(hm)]
    if len(valid) > 2:
        arr   = np.array(valid)
        r     = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
        fig.add_annotation(
            x=0.05, y=0.95, xref="x2 domain", yref="y2 domain",
            text=f"r = {r:.4f}", showarrow=False,
            bgcolor="white", bordercolor="gray", font_size=11,
        )

    fig.update_xaxes(title_text=f"LM flux ({unit})", row=1, col=2)
    fig.update_yaxes(title_text=f"HM flux ({unit})", row=1, col=2)
    fig.update_layout(
        template="none",
        title=f"<b>Model comparison – {sensor}</b>",
        height=480,
        legend=dict(orientation="h", y=-0.20),
    )
    return fig
