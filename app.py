"""Streamlit app – Respiration Chamber Analysis / pyFlux."""

import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st

import chamber_analysis as ca
from chamber_analysis import plots
from chamber_analysis.pyflux import analyze_directory, GAS_META

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chamber Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────
for _k, _v in [
    ("all_dfs",      {}),
    ("results",      []),
    ("summary",      pd.DataFrame()),
    ("loaded_names", set()),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Chamber Parameters")

    col_v, col_a = st.columns(2)
    volume_L = col_v.number_input("Volume (L)", value=13.57,  min_value=0.1,   step=0.1)
    area_m2  = col_a.number_input("Area (m²)", value=0.04524, min_value=0.0001,
                                  format="%.5f", step=0.001)

    st.divider()

    t_auto = st.toggle("Temperature: read from data", value=True)
    if t_auto:
        t_val = "auto"
        st.caption("Uses mean of `Temperature intern (degC)`")
    else:
        t_val = float(st.number_input("Temperature (°C)", value=20.0, step=0.5))

    p_auto = st.toggle("Pressure: read from data", value=True)
    if p_auto:
        p_val = "auto"
        st.caption("Uses mean of `Pressure intern (mbar)`")
    else:
        p_val = float(st.number_input("Pressure (mbar)", value=1013.25, step=1.0))

    window_frac = st.slider(
        "Analysis window fraction", 0.5, 1.0, 0.8, 0.05,
        help="Fraction of measurement duration for the sliding best-R² window.",
    )
    window_undercut = st.slider(
        "Max window reduction", 0.0, 0.5, 0.3, 0.05,
        help="How far the window may shrink below the target fraction to find "
             "a more linear segment (e.g. 0.3 = up to 30 % smaller). "
             "Useful for saturating fluxes where the plateau degrades linearity.",
    )
    g_factor_max = st.slider(
        "g-factor max", 1.0, 10.0, 4.0, 0.5,
        help="Max |HM flux / LM flux| ratio before falling back to LM. "
             "Raise for strongly saturating fluxes.",
    )

    st.divider()
    run_btn = st.button("Run Analysis", type="primary", width="stretch")

    st.divider()
    st.subheader("Active Sensors")
    active_sensors = [
        s for s in GAS_META
        if st.checkbox(s, value=True, key=f"s_{s}")
    ]

    # Quick stats after analysis
    if st.session_state.results:
        s = st.session_state.summary
        st.divider()
        st.caption(f"**{len(s)} results** – {len(s['filename'].unique())} files × {len(s['sensor'].unique())} sensors")
        good = (s["quality"] == "good").sum()
        st.caption(f"{good}/{len(s)} results quality: good")


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("Respiration Chamber Analysis")
st.caption("Upload CSV files · configure chamber · calculate GHG fluxes with pyFlux (LM + HM)")

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload chamber CSV files",
    type="csv",
    accept_multiple_files=True,
    help="Semicolon-delimited CSV files produced by the chamber logger.",
)

if uploaded:
    current_names = {uf.name for uf in uploaded}
    if current_names != st.session_state.loaded_names:
        new_dfs = {}
        errors  = []
        with st.spinner(f"Parsing {len(uploaded)} file(s)…"):
            for uf in uploaded:
                try:
                    df = ca.load_csv(uf)
                    if df.empty:
                        errors.append(f"{uf.name}: empty after parsing")
                    else:
                        new_dfs[uf.name] = df
                except Exception as exc:
                    errors.append(f"{uf.name}: {exc}")
        for msg in errors:
            st.warning(msg)
        st.session_state.all_dfs      = new_dfs
        st.session_state.loaded_names = current_names
        st.session_state.results      = []
        st.session_state.summary      = pd.DataFrame()

all_dfs = st.session_state.all_dfs

if not all_dfs:
    st.info("Upload one or more CSV files to begin.")
    st.stop()

# ── File overview expander ─────────────────────────────────────────────────────
with st.expander(f"{len(all_dfs)} file(s) loaded", expanded=False):
    rows = []
    for fname, df in all_dfs.items():
        m = ca.extract_metadata(df, fname)
        rows.append({
            "File":           fname,
            "Rows":           len(df),
            "Duration (min)": round(df["time_min"].max(), 1),
            "Start":          str(m.timestamp_start),
            "GPS lat":        m.gps_latitude,
            "GPS lon":        m.gps_longitude,
            "T intern (°C)":  m.temperature_intern_C,
            "P intern (mbar)":m.pressure_intern_mbar,
            "Battery (%)":    m.battery_level_pct,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if run_btn:
    if not active_sensors:
        st.error("Select at least one sensor in the sidebar.")
    else:
        params = ca.ChamberParams(
            volume_L=volume_L, area_m2=area_m2,
            temperature_C=t_val, pressure_mbar=p_val,
            window_frac=window_frac,
            window_undercut=window_undercut,
            g_factor_max=g_factor_max,
        )
        with st.spinner("Running pyFlux (LM + HM for each sensor × file)…"):
            results, summary = analyze_directory(all_dfs, params, sensors=active_sensors)
        st.session_state.results = results
        st.session_state.summary = summary
        if summary.empty:
            st.warning(
                "No recognized sensor columns found in the uploaded file(s). "
                "Check that column names match the supported sensors "
                "(see the sidebar **Active Sensors** list)."
            )
        else:
            st.success(
                f"Done: {len(results)} results  |  "
                f"{(summary['quality']=='good').sum()} good  |  "
                f"{(summary['model']=='HM').sum()} HM selected"
            )

results = st.session_state.results
summary = st.session_state.summary

if not results:
    st.info("Set parameters in the sidebar and click **Run Analysis**.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
(tab_ts, tab_reg, tab_flux, tab_overview,
 tab_cmp, tab_bat, tab_table, tab_export, tab_methods) = st.tabs([
    "Time Series", "Regression", "Flux",
    "Overview", "LM vs HM", "Battery",
    "Results Table", "Export", "Methods",
])

# ── Time Series ────────────────────────────────────────────────────────────────
with tab_ts:
    ts_fname = st.selectbox("File", sorted(all_dfs.keys()), key="ts_file")
    ts_df    = all_dfs[ts_fname]
    ts_cols  = [s for s in active_sensors if s in ts_df.columns]

    # Metadata strip
    m = ca.extract_metadata(ts_df, ts_fname)
    mc = st.columns(5)
    mc[0].metric("Duration (min)", f"{ts_df['time_min'].max():.1f}")
    mc[1].metric("T intern (°C)",  f"{m.temperature_intern_C:.1f}" if m.temperature_intern_C else "—")
    mc[2].metric("P intern (mbar)",f"{m.pressure_intern_mbar:.1f}" if m.pressure_intern_mbar else "—")
    mc[3].metric("GPS lat",        f"{m.gps_latitude:.5f}"          if m.gps_latitude         else "—")
    mc[4].metric("Battery (%)",    f"{m.battery_level_pct:.0f}"     if m.battery_level_pct    else "—")

    st.plotly_chart(
        plots.time_series_fig(ts_df, ts_cols, title=ts_fname),
        width="stretch",
    )

# ── Regression ─────────────────────────────────────────────────────────────────
with tab_reg:
    # @st.fragment prevents a full-page rerun when selectors change,
    # which would otherwise reset the active tab back to the first one.
    @st.fragment
    def _regression_tab():
        _results  = st.session_state.results
        _all_dfs  = st.session_state.all_dfs
        _sensors  = [s for s in GAS_META if st.session_state.get(f"s_{s}", True)]

        rc1, rc2   = st.columns(2)
        reg_fname  = rc1.selectbox("File",   sorted(_all_dfs.keys()), key="reg_file")
        reg_sensor = rc2.selectbox("Sensor", _sensors,                 key="reg_sensor")

        r = next(
            (x for x in _results if x.filename == reg_fname and x.sensor == reg_sensor),
            None,
        )
        if r is None:
            st.warning("No result for this combination — sensor may have been inactive or had too few points.")
            return

        st.plotly_chart(
            plots.regression_fig(r, _all_dfs[reg_fname]),
            width="stretch",
        )

        rf     = GAS_META[reg_sensor]["report_factor"]
        lm     = r.flux_result.lm
        hm     = r.flux_result.hm
        unit   = r.report_unit
        chosen = r.flux_result.model

        lm_col, hm_col = st.columns(2)

        with lm_col:
            st.subheader(f"{'★ ' if chosen == 'LM' else ''}Linear Model (LM)")
            mc = st.columns(3)
            mc[0].metric("Flux", f"{lm.flux * rf:.5f}")
            mc[1].metric("SE",   f"±{lm.flux_se * rf:.5f}")
            mc[2].metric("Unit", unit)
            mc2 = st.columns(4)
            mc2[0].metric("R²",   f"{lm.r_squared:.4f}")
            mc2[1].metric("p",    f"{lm.p_value:.4f}")
            mc2[2].metric("AICc", f"{lm.aicc:.1f}")
            mc2[3].metric("RMSE", f"{lm.rmse:.4f}")

        with hm_col:
            st.subheader(f"{'★ ' if chosen == 'HM' else ''}Hutchinson-Mosier (HM)")
            if hm and hm.converged:
                mc = st.columns(3)
                mc[0].metric("Flux",  f"{hm.flux * rf:.5f}")
                mc[1].metric("SE",    f"±{hm.flux_se * rf:.5f}")
                mc[2].metric("Unit",  unit)
                mc2 = st.columns(4)
                mc2[0].metric("R²",   f"{hm.r_squared:.4f}")
                mc2[1].metric("AICc", f"{hm.aicc:.1f}")
                mc2[2].metric("RMSE", f"{hm.rmse:.4f}")
                mc2[3].metric("g",    f"{r.flux_result.g_factor:.3f}")
                mc3 = st.columns(3)
                mc3[0].metric("C₀ (ppm)", f"{hm.C0:.2f}")
                mc3[1].metric("C∞ (ppm)", f"{hm.Cinf:.2f}")
                mc3[2].metric("κ (s⁻¹)",  f"{hm.kappa:.5f}")
            else:
                st.info("HM did not converge for this measurement.")

    _regression_tab()

# ── Flux bar ───────────────────────────────────────────────────────────────────
with tab_flux:
    flux_sensor = st.selectbox("Sensor", active_sensors, key="flux_sensor")
    st.plotly_chart(
        plots.flux_bar_fig(results, flux_sensor),
        width="stretch",
    )

# ── Overview ───────────────────────────────────────────────────────────────────
with tab_overview:
    st.plotly_chart(
        plots.overview_fig(results, all_dfs),
        width="stretch",
    )

# ── LM vs HM ──────────────────────────────────────────────────────────────────
with tab_cmp:
    cmp_sensor = st.selectbox("Sensor", active_sensors, key="cmp_sensor")
    st.plotly_chart(
        plots.model_comparison_fig(results, cmp_sensor),
        width="stretch",
    )

    sub = summary[summary["sensor"] == cmp_sensor]
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Model selection")
        st.dataframe(
            sub.groupby("model").size().rename("count").reset_index(),
            hide_index=True, width="stretch",
        )
    with cc2:
        st.subheader("Quality flags")
        st.dataframe(
            sub.groupby("quality").size().rename("count").reset_index(),
            hide_index=True, width="stretch",
        )

# ── Battery ────────────────────────────────────────────────────────────────────
with tab_bat:
    st.plotly_chart(plots.battery_fig(all_dfs), width="stretch")

# ── Results Table ──────────────────────────────────────────────────────────────
with tab_table:
    fc1, fc2, fc3 = st.columns(3)
    flt_sensor  = fc1.selectbox("Sensor",  ["All"] + active_sensors,
                                key="tbl_sensor")
    flt_model   = fc2.selectbox("Model",   ["All", "LM", "HM"],
                                key="tbl_model")
    flt_quality = fc3.selectbox("Quality", ["All", "good", "below_mdf", "below_detection"],
                                key="tbl_quality")

    tbl = summary.copy()
    if flt_sensor  != "All": tbl = tbl[tbl["sensor"]  == flt_sensor]
    if flt_model   != "All": tbl = tbl[tbl["model"]   == flt_model]
    if flt_quality != "All": tbl = tbl[tbl["quality"] == flt_quality]

    st.caption(f"{len(tbl)} of {len(summary)} rows")
    st.dataframe(tbl, width="stretch", hide_index=True)

# ── Export ─────────────────────────────────────────────────────────────────────
with tab_export:
    ts_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname_out = f"Flux_pyFlux_{ts_str}.csv"

    buf = io.StringIO()
    summary.to_csv(buf, index=False, sep=";", decimal=",")
    csv_bytes = ("﻿" + buf.getvalue()).encode("utf-8")  # UTF-8 BOM for Excel

    st.download_button(
        label=f"Download {fname_out}",
        data=csv_bytes,
        file_name=fname_out,
        mime="text/csv",
        type="primary",
    )
    st.caption(
        f"{len(summary)} rows × {len(summary.columns)} columns  |  "
        "semicolon-delimited · comma decimal · UTF-8 BOM"
    )

    st.divider()
    st.subheader("Preview")
    st.dataframe(summary.head(10), width="stretch", hide_index=True)

# ── Methods ────────────────────────────────────────────────────────────────────
with tab_methods:
    _methods_path = __file__.replace("app.py", "METHODS.md")
    try:
        with open(_methods_path, encoding="utf-8") as _f:
            st.markdown(_f.read())
    except FileNotFoundError:
        st.error("METHODS.md not found next to app.py.")
